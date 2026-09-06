"""Package-level exception types.

Leaf module — no dependencies on the rest of the package — so any layer
(profile, handlers, surface) can raise these without import cycles. Loader
errors stay in `chumak.loader`; they are specific to that machinery.
"""

from __future__ import annotations


class ProfileCapabilityError(ValueError):
    """A profile was asked for something its handler cannot provide.

    Handler capabilities are deliberately uneven (a subprocess profile has no
    chat model to hand out; a langchain profile has no command to run). This
    is the typed signal for "valid profile, wrong handler for the request".
    Subclasses `ValueError` so existing `except ValueError` guards still hold.
    """
