import math

def naive_round(xval_rel, model):
    """Round integer variables to nearest integer."""
    # Assign solution values
    for k, vals in xval_rel.items():
        # Assign initial solution
        for i in vals:
            # Round integer variables
            if model.vars[k][i].is_integer():
                model.vars[k][i].value = round(vals[i])
            # Assign continuous variables
            else:
                model.vars[k][i].value = vals[i]
    xval, objval = model.get_val()
    return xval, objval

def floor_round(xval_rel, model):
    """Round integer variables down (floor)."""
    # Assign solution values
    for k, vals in xval_rel.items():
        # Assign initial solution
        for i in vals:
            # Round integer variables
            if model.vars[k][i].is_integer():
                model.vars[k][i].value = math.floor(vals[i])
            # Assign continuous variables
            else:
                model.vars[k][i].value = vals[i]
    xval, objval = model.get_val()
    return xval, objval
