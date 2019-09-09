# Macros for building CM code.
    
def cm_is_configured():
    """Returns true if CM was enabled during the configure process."""
    return %{cm_is_configured}

def if_cm_is_configured(x):
    """Tests if CM was enabled during the configure process."""
    if cm_is_configured():
        return x
    return []
