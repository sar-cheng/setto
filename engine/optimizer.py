from engine.plan import Filter, Plan, Scan, Select


def _optimize_scan(
    plan: Scan, required_columns: set[str] | None = None
) -> Plan:
    needed_by_parent = (
        required_columns if required_columns is not None else set(plan.schema)
    )
    # preserve order
    selected = [col for col in plan.schema if col in needed_by_parent]

    return Scan(plan.path, columns=selected, batch_size=plan.batch_size)


def _optimize_select(
    plan: Select, required_columns: set[str] | None = None
) -> Plan:
    needed_by_parent = (
        required_columns if required_columns is not None else set(plan.cols)
    )
    selected = [col for col in plan.cols if col in needed_by_parent]

    return Select(selected, optimize(plan.child, set(selected)))


def _optimize_filter(
    plan: Filter, required_columns: set[str] | None = None
) -> Plan:
    if isinstance(plan.child, Select):
        if plan.predicate.columns() <= set(plan.child.cols):
            # Predicate pushdown
            pushed = Select(
                plan.child.cols,
                child=Filter(plan.predicate, plan.child.child),
            )
            # After rewriting the tree, new optimization opportunities may appear
            # e.g. projection pushdown below the new Filter
            return optimize(pushed, required_columns)

    needed_by_parent = (
        required_columns if required_columns is not None else set(plan.schema)
    )

    child_required = needed_by_parent | plan.predicate.columns()

    return Filter(plan.predicate, optimize(plan.child, child_required))


def optimize(plan: Plan, required_columns: set[str] | None = None) -> Plan:
    match plan:
        case Scan():
            return _optimize_scan(plan, required_columns)
        case Select():
            return _optimize_select(plan, required_columns)
        case Filter():
            return _optimize_filter(plan, required_columns)

    raise TypeError(f"Unknown plan node: {type(plan).__name__}")
