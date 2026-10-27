//! `compile_database_scans`: the last optimizer pass for database scans.
//!
//! It sees each scan's final projection, predicate and slice, translates the predicate into
//! dialect-neutral SQL, resolves partitions and renders the SQL of every partition into the
//! scan config. Workers then only run finished SQL.
//!
//! A pushed conjunct may make the database return more rows than Polars keeps, never fewer.
//! Each translated expression is classed as exact or superset; anything else is not pushed.

use std::sync::Arc;

use polars_core::prelude::*;
use polars_io::database::dialect::{DialectName, StringSemantics};
use polars_io::database::sql::{SqlBinOp, SqlExpr, SqlLiteral};
use polars_io::database::{DatabaseScanConfig, PushedFilter, ScanPushdown, compile_scan};
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_path::PlRefPath;
use polars_utils::slice_enum::Slice;
use polars_utils::unitvec;

use crate::dsl::{FileScanIR, ScanSources};
#[cfg(feature = "strings")]
use crate::plans::aexpr::IRStringFunction;
use crate::plans::aexpr::{AExpr, IRBooleanFunction, IRFunctionExpr, MintermIter};
use crate::plans::{IR, LiteralValue};
use crate::prelude::Operator;

pub(super) fn compile_database_scans(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<()> {
    let mut stack = unitvec![root];

    while let Some(node) = stack.pop() {
        ir_arena.get(node).copy_inputs(&mut stack);

        let IR::Scan {
            sources,
            scan_type,
            predicate,
            unified_scan_args,
            ..
        } = ir_arena.get_mut(node)
        else {
            continue;
        };
        let FileScanIR::Database { config } = scan_type.as_mut() else {
            continue;
        };

        // The scan's row index and slice apply to the rows before the predicate, so SQL may
        // only filter when neither is set.
        let args = &**unified_scan_args;
        let pre_filter_ops = args.row_index.is_some() || args.pre_slice.is_some();
        // Without pushdown the filters are only counted.
        let translate = !pre_filter_ops && config.pushdown_disabled.is_none();

        let mut pushdown = ScanPushdown {
            columns: args.projection.as_ref().map(|p| {
                p.iter()
                    .filter(|c| config.schema.contains(c))
                    .cloned()
                    .collect()
            }),
            ..Default::default()
        };

        if let Some(predicate) = predicate.as_ref() {
            let translator = Translator {
                config,
                arena: expr_arena,
            };
            for term in MintermIter::new(predicate.node(), expr_arena) {
                pushdown.total_filters += 1;
                if !translate {
                    continue;
                }
                if let Some((expr, class)) = translator.translate(term) {
                    pushdown.filters.push(PushedFilter {
                        expr,
                        exact: class == Class::Exact,
                    });
                }
            }
        }

        if predicate.is_none()
            && let Some(Slice::Positive { offset, len }) = args.pre_slice
        {
            pushdown.limit = Some((offset + len) as u64);
        }

        let target = args.resolve_heavy_sources.map(|n| n.get() as usize);
        let compiled = compile_scan(config, pushdown, target)?;
        let n = compiled.partitions.len();

        let mut new_config = DatabaseScanConfig::clone(config);
        new_config.compiled = Some(Arc::new(compiled));
        *config = Arc::new(new_config);

        // Source ids are absolute indices into the compiled partitions.
        *sources = ScanSources::Paths(polars_buffer::Buffer::from_owner(
            (0..n)
                .map(|i| PlRefPath::new(i.to_string()))
                .collect::<Vec<_>>(),
        ));
    }

    Ok(())
}

/// How a translated expression relates to the Polars predicate `P`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    /// `P(row)` implies the SQL is TRUE: usable as a pre-filter that Polars re-applies.
    Superset,
    /// The SQL is TRUE exactly when `P(row)` is true.
    Exact,
}

enum Operand<'a> {
    Column(&'a PlSmallStr, &'a DataType),
    Literal(&'a LiteralValue),
}

struct Translator<'a> {
    config: &'a DatabaseScanConfig,
    arena: &'a Arena<AExpr>,
}

impl Translator<'_> {
    fn dialect(&self) -> DialectName {
        self.config.dialect
    }

    fn strings(&self, column: &str) -> StringSemantics {
        self.config
            .column(column)
            .map_or_else(|| self.dialect().default_string_semantics(), |c| c.strings)
    }

    /// Whether the driver binds `params=` into the SQL: a `%` in a pushed literal would then
    /// be read as a placeholder by `%s`-style drivers, so such literals aren't pushed.
    fn binds_params(&self) -> bool {
        self.config.options.connection.has_query_params()
    }

    /// MySQL literals are written for the session setup: backslash escapes on
    /// (`NO_BACKSLASH_ESCAPES` removed) and a UTC time zone. Where it doesn't run, the
    /// server may read them otherwise, so such literals aren't pushed.
    fn mysql_without_setup(&self) -> bool {
        self.dialect() == DialectName::Mysql && !self.config.session_setup_runs()
    }

    /// A literal that can be written into the SQL (see `binds_params`, `mysql_without_setup`).
    fn writable(&self, lit: &SqlLiteral) -> bool {
        let Some(s) = lit.as_str() else {
            return true;
        };
        let placeholder = self.binds_params() && s.contains('%');
        let escape = self.mysql_without_setup() && s.contains('\\');
        !(placeholder || escape)
    }

    /// Whether predicates on the column can be pushed at all (see `ColumnMeta::comparable`).
    fn comparable(&self, column: &str) -> bool {
        self.config.column(column).is_none_or(|c| c.comparable)
    }

    /// A boolean expression as SQL, with its class. `None` when it can't be pushed.
    fn translate(&self, node: Node) -> Option<(SqlExpr, Class)> {
        match self.arena.get(node) {
            AExpr::BinaryExpr { left, op, right } => match op {
                Operator::And | Operator::LogicalAnd => {
                    match (self.translate(*left), self.translate(*right)) {
                        (Some((l, lc)), Some((r, rc))) => {
                            Some((SqlExpr::binary(l, SqlBinOp::And, r), lc.min(rc)))
                        },
                        // Dropping one side of an AND keeps every row the AND keeps.
                        (Some((e, _)), None) | (None, Some((e, _))) => Some((e, Class::Superset)),
                        (None, None) => None,
                    }
                },
                Operator::Or | Operator::LogicalOr => {
                    let (l, lc) = self.translate(*left)?;
                    let (r, rc) = self.translate(*right)?;
                    Some((SqlExpr::binary(l, SqlBinOp::Or, r), lc.min(rc)))
                },
                op => self.comparison(*left, *op, *right),
            },
            AExpr::Column(name) => {
                // A boolean column used as the predicate.
                let dtype = self.config.schema.get(name)?;
                if !self.comparable(name) {
                    return None;
                }
                (dtype == &DataType::Boolean && self.dialect().supports_boolean()).then(|| {
                    (
                        SqlExpr::binary(
                            SqlExpr::col(name.clone()),
                            SqlBinOp::Eq,
                            SqlExpr::lit(SqlLiteral::Bool(true)),
                        ),
                        Class::Exact,
                    )
                })
            },
            AExpr::Function {
                input, function, ..
            } => self.function(input, function),
            _ => None,
        }
    }

    fn function(
        &self,
        input: &[crate::plans::ExprIR],
        function: &IRFunctionExpr,
    ) -> Option<(SqlExpr, Class)> {
        match function {
            IRFunctionExpr::Boolean(f) => match f {
                IRBooleanFunction::Not => {
                    // The negation of a superset is a subset: only exact expressions negate.
                    let (inner, class) = self.translate(input.first()?.node())?;
                    (class == Class::Exact).then(|| (SqlExpr::Not(Box::new(inner)), Class::Exact))
                },
                IRBooleanFunction::IsNull | IRBooleanFunction::IsNotNull => {
                    let (name, _) = self.column(input.first()?.node())?;
                    Some((
                        SqlExpr::IsNull(
                            Box::new(SqlExpr::col(name.clone())),
                            matches!(f, IRBooleanFunction::IsNotNull),
                        ),
                        Class::Exact,
                    ))
                },
                #[cfg(feature = "is_between")]
                IRBooleanFunction::IsBetween { closed } => {
                    use crate::prelude::ClosedInterval;
                    let [e, lo, hi] = input else { return None };
                    let (lo_op, hi_op) = match closed {
                        ClosedInterval::Both => (Operator::GtEq, Operator::LtEq),
                        ClosedInterval::Left => (Operator::GtEq, Operator::Lt),
                        ClosedInterval::Right => (Operator::Gt, Operator::LtEq),
                        ClosedInterval::None => (Operator::Gt, Operator::Lt),
                    };
                    let (l, lc) = self.comparison(e.node(), lo_op, lo.node())?;
                    let (r, rc) = self.comparison(e.node(), hi_op, hi.node())?;
                    Some((SqlExpr::binary(l, SqlBinOp::And, r), lc.min(rc)))
                },
                #[cfg(feature = "is_in")]
                IRBooleanFunction::IsIn {
                    nulls_equal,
                    needle_cast: None,
                } => self.is_in(input, *nulls_equal),
                _ => None,
            },
            #[cfg(feature = "strings")]
            IRFunctionExpr::StringExpr(f) => {
                let [e, pat] = input else { return None };
                let (name, dtype) = self.column(e.node())?;
                let AExpr::Literal(lv) = self.arena.get(pat.node()) else {
                    return None;
                };
                // `LIKE` patterns hold `%`, which `%s`-style drivers read as a placeholder,
                // and `ESCAPE '\\'` relies on MySQL's session setup.
                if !dtype.is_string() || self.binds_params() || self.mysql_without_setup() {
                    return None;
                }
                let pat = lv.extract_str()?;
                let (prefix, suffix) = match f {
                    IRStringFunction::StartsWith => ("", "%"),
                    IRStringFunction::EndsWith => ("%", ""),
                    #[cfg(feature = "regex")]
                    IRStringFunction::Contains { literal: true, .. } => ("%", "%"),
                    _ => return None,
                };
                let strings = self.strings(name);
                if strings.empty_is_null && pat.is_empty() {
                    return None;
                }
                let pattern = format!("{prefix}{}{suffix}", self.escape_like(pat));
                let class = if strings.equality_exact && !self.dialect().like_case_insensitive() {
                    Class::Exact
                } else {
                    Class::Superset
                };
                Some((
                    SqlExpr::Like {
                        expr: Box::new(SqlExpr::col(name.clone())),
                        pattern,
                        escape: '\\',
                    },
                    class,
                ))
            },
            _ => None,
        }
    }

    /// Whether a date or timestamp literal is within what the dialect's types hold.
    fn temporal_in_range(&self, av: &AnyValue<'_>) -> bool {
        let (lo, hi) = self.dialect().temporal_literal_days();
        let days = match av {
            AnyValue::Date(d) => *d as i64,
            AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _) => {
                let per_day = match tu {
                    TimeUnit::Nanoseconds => 86_400_000_000_000,
                    TimeUnit::Microseconds => 86_400_000_000,
                    TimeUnit::Milliseconds => 86_400_000,
                };
                v.div_euclid(per_day)
            },
            _ => return true,
        };
        (lo..=hi).contains(&days)
    }

    fn escape_like(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            let special = matches!(c, '%' | '_' | '\\')
                || (c == '[' && matches!(self.dialect(), DialectName::Mssql));
            if special {
                out.push('\\');
            }
            out.push(c);
        }
        out
    }

    #[cfg(feature = "is_in")]
    fn is_in(&self, input: &[crate::plans::ExprIR], nulls_equal: bool) -> Option<(SqlExpr, Class)> {
        let [e, list] = input else { return None };
        let (name, dtype) = self.column(e.node())?;
        // The haystack is a list scalar, or a Series (possibly of one list).
        let values = match self.arena.get(list.node()) {
            AExpr::Literal(LiteralValue::Series(s)) => match s.dtype() {
                DataType::List(_) if s.len() == 1 => s.list().ok()?.get_as_series(0)?,
                _ => (**s).clone(),
            },
            AExpr::Literal(LiteralValue::Scalar(sc)) => match sc.value() {
                AnyValue::List(s) => s.clone(),
                _ => return None,
            },
            _ => return None,
        };

        let mut has_null = false;
        let mut lits = Vec::with_capacity(values.len());
        for av in values.iter() {
            if av.is_null() {
                has_null = true;
                continue;
            }
            if !self.temporal_in_range(&av) {
                return None;
            }
            let lit = SqlLiteral::from_any_value(&av, dtype)?;
            if !self.writable(&lit) {
                return None;
            }
            if let Some(s) = lit.as_str()
                && s.is_empty()
                && self.strings(name).empty_is_null
            {
                return None;
            }
            lits.push(lit);
        }
        if lits.is_empty() {
            return None;
        }
        let mut class = self.leaf_class(name, dtype, Operator::Eq)?;

        let col = || Box::new(SqlExpr::col(name.clone()));
        let mut expr = if lits.len() <= self.dialect().max_in_list() {
            SqlExpr::InList(col(), lits, false)
        } else {
            // Too long for one statement: a range that contains every value.
            if self.leaf_class(name, dtype, Operator::Lt).is_none() {
                return None;
            }
            let mut sorted = values.drop_nulls().sort(Default::default()).ok()?;
            sorted = sorted.rechunk();
            let lo = SqlLiteral::from_any_value(&sorted.get(0).ok()?, dtype)?;
            let hi = SqlLiteral::from_any_value(&sorted.get(sorted.len() - 1).ok()?, dtype)?;
            class = Class::Superset;
            SqlExpr::Between(
                col(),
                Box::new(SqlExpr::lit(lo)),
                Box::new(SqlExpr::lit(hi)),
            )
        };
        if nulls_equal {
            // With nulls_equal the result is never NULL, so the SQL must not be either.
            expr = if has_null {
                SqlExpr::binary(expr, SqlBinOp::Or, SqlExpr::IsNull(col(), false))
            } else {
                SqlExpr::binary(SqlExpr::IsNull(col(), true), SqlBinOp::And, expr)
            };
        }
        Some((expr, class))
    }

    fn column(&self, node: Node) -> Option<(&PlSmallStr, &DataType)> {
        match self.operand(node)? {
            Operand::Column(name, dtype) => Some((name, dtype)),
            Operand::Literal(_) => None,
        }
    }

    fn operand(&self, node: Node) -> Option<Operand<'_>> {
        match self.arena.get(node) {
            AExpr::Column(name) => {
                let (name, dtype) = self.config.schema.get_full(name).map(|(_, n, d)| (n, d))?;
                self.comparable(name)
                    .then_some(Operand::Column(name, dtype))
            },
            AExpr::Literal(lv) if lv.is_scalar() => Some(Operand::Literal(lv)),
            // A lossless widening of an integer column compares like the column itself.
            AExpr::Cast { expr, dtype, .. } => match self.operand(*expr)? {
                Operand::Column(name, col_dtype)
                    if col_dtype.is_integer()
                        && dtype.is_integer()
                        && dtype.is_signed_integer() >= col_dtype.is_signed_integer()
                        && widens(col_dtype, dtype) =>
                {
                    Some(Operand::Column(name, col_dtype))
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// The class of `column <op> literal`, from the column's type and semantics.
    fn leaf_class(&self, name: &str, dtype: &DataType, op: Operator) -> Option<Class> {
        let ordering = matches!(
            op,
            Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
        );
        match dtype {
            dt if dt.is_integer() => Some(Class::Exact),
            DataType::Boolean => self.dialect().supports_boolean().then_some(Class::Exact),
            // SQL Server's legacy `datetime` holds 1/300 s ticks, which the microsecond
            // literal doesn't match exactly.
            DataType::Datetime(_, _)
                if self.dialect() == DialectName::Mssql
                    && self
                        .config
                        .column(name)
                        .and_then(|c| c.db_type.as_deref())
                        .is_some_and(|t| {
                            matches!(
                                t.trim().to_ascii_lowercase().as_str(),
                                "datetime" | "smalldatetime"
                            )
                        }) =>
            {
                None
            },
            // ClickHouse parses a literal without a zone in the server's time zone, which
            // Polars doesn't know.
            DataType::Datetime(_, None) if self.dialect() == DialectName::Clickhouse => None,
            // A MySQL `TIMESTAMP` literal is read in the session time zone, UTC only after
            // the session setup.
            DataType::Datetime(_, Some(_)) if self.mysql_without_setup() => None,
            DataType::Date | DataType::Datetime(_, _) => self
                .dialect()
                .temporal_comparisons_exact()
                .then_some(Class::Exact),
            DataType::Time => self.dialect().supports_time().then_some(Class::Exact),
            // An unconstrained `numeric` / `NUMBER` is read rounded to this scale, so the
            // database compares digits Polars doesn't see.
            #[cfg(feature = "dtype-decimal")]
            dt if *dt == polars_io::database::schema::UNCONSTRAINED_DECIMAL => None,
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_, _) => self
                .dialect()
                .decimal_comparisons_exact()
                .then_some(Class::Exact),
            // Oracle `BINARY_DOUBLE` keeps -0.0 apart from 0.0, which Polars treats as equal;
            // `NUMBER` and `FLOAT` hold decimals that the f64 Polars reads only approximates.
            // No Oracle float comparison is pushed.
            dt if dt.is_float() && self.dialect() == DialectName::Oracle => None,
            dt if dt.is_float() => self
                .dialect()
                .float_comparisons_exact()
                .then_some(Class::Exact),
            DataType::String => {
                let s = self.strings(name);
                match op {
                    _ if ordering => s.ordering_exact.then_some(Class::Exact),
                    Operator::Eq | Operator::EqValidity => Some(if s.equality_exact {
                        Class::Exact
                    } else {
                        Class::Superset
                    }),
                    // `<>` under a case-insensitive collation drops rows Polars keeps.
                    _ => s.equality_exact.then_some(Class::Exact),
                }
            },
            _ => None,
        }
    }

    fn comparison(&self, left: Node, op: Operator, right: Node) -> Option<(SqlExpr, Class)> {
        let (l, r) = (self.operand(left)?, self.operand(right)?);
        let (name, dtype, lv, op) = match (l, r) {
            (Operand::Column(n, d), Operand::Literal(lv)) => (n, d, lv, op),
            (Operand::Literal(lv), Operand::Column(n, d)) => (n, d, lv, op.swap_operands()?),
            (Operand::Column(a, ad), Operand::Column(b, bd)) => {
                return self.column_comparison(a, ad, op, b, bd);
            },
            _ => return None,
        };

        let bin_op = match op {
            Operator::Eq | Operator::EqValidity => SqlBinOp::Eq,
            Operator::NotEq | Operator::NotEqValidity => SqlBinOp::NotEq,
            Operator::Lt => SqlBinOp::Lt,
            Operator::LtEq => SqlBinOp::LtEq,
            Operator::Gt => SqlBinOp::Gt,
            Operator::GtEq => SqlBinOp::GtEq,
            _ => return None,
        };
        let col = || SqlExpr::col(name.clone());

        let av = lv.to_any_value()?;
        if av.is_null() {
            return match op {
                // `eq_missing(None)` is `is_null`.
                Operator::EqValidity => {
                    Some((SqlExpr::IsNull(Box::new(col()), false), Class::Exact))
                },
                Operator::NotEqValidity => {
                    Some((SqlExpr::IsNull(Box::new(col()), true), Class::Exact))
                },
                // A comparison with NULL keeps no rows in either system.
                _ => None,
            };
        }

        let lit = SqlLiteral::from_any_value(&av, dtype)?;
        if !self.writable(&lit) || !self.temporal_in_range(&av) {
            return None;
        }
        if let Some(s) = lit.as_str()
            && s.is_empty()
            && self.strings(name).empty_is_null
        {
            return None;
        }
        if matches!(lit, SqlLiteral::Bool(_)) && !self.dialect().supports_boolean() {
            return None;
        }
        let class = self.leaf_class(name, dtype, op)?;
        let mut expr = SqlExpr::binary(col(), bin_op, SqlExpr::lit(lit));
        // `eq_missing` / `ne_missing` are never NULL, so their SQL must not be either: under
        // a NOT, a NULL would drop rows that Polars keeps.
        if op == Operator::NotEqValidity {
            // NULL <> x is NULL; ne_missing(NULL, x) is true.
            expr = SqlExpr::binary(SqlExpr::IsNull(Box::new(col()), false), SqlBinOp::Or, expr);
        } else if op == Operator::EqValidity {
            // NULL = x is NULL; eq_missing(NULL, x) is false.
            expr = SqlExpr::binary(SqlExpr::IsNull(Box::new(col()), true), SqlBinOp::And, expr);
        }
        Some((expr, class))
    }

    /// `a <op> b` on two columns: only for types without collation or NaN concerns.
    fn column_comparison(
        &self,
        a: &PlSmallStr,
        ad: &DataType,
        op: Operator,
        b: &PlSmallStr,
        bd: &DataType,
    ) -> Option<(SqlExpr, Class)> {
        let date_exact =
            matches!(ad, DataType::Date) && self.dialect().temporal_comparisons_exact();
        if ad != bd || !(ad.is_integer() || date_exact) {
            return None;
        }
        let bin_op = match op {
            Operator::Eq => SqlBinOp::Eq,
            Operator::NotEq => SqlBinOp::NotEq,
            Operator::Lt => SqlBinOp::Lt,
            Operator::LtEq => SqlBinOp::LtEq,
            Operator::Gt => SqlBinOp::Gt,
            Operator::GtEq => SqlBinOp::GtEq,
            _ => return None,
        };
        Some((
            SqlExpr::binary(SqlExpr::col(a.clone()), bin_op, SqlExpr::col(b.clone())),
            Class::Exact,
        ))
    }
}

/// Whether every value of `from` is representable in `to` (both integers).
fn widens(from: &DataType, to: &DataType) -> bool {
    let bits = |d: &DataType| -> Option<u32> {
        Some(match d {
            DataType::Int8 | DataType::UInt8 => 8,
            DataType::Int16 | DataType::UInt16 => 16,
            DataType::Int32 | DataType::UInt32 => 32,
            DataType::Int64 | DataType::UInt64 => 64,
            DataType::Int128 | DataType::UInt128 => 128,
            _ => return None,
        })
    };
    match (bits(from), bits(to)) {
        (Some(f), Some(t)) => {
            if from.is_signed_integer() == to.is_signed_integer() {
                t >= f
            } else {
                // unsigned -> signed needs one more bit.
                to.is_signed_integer() && t > f
            }
        },
        _ => false,
    }
}
