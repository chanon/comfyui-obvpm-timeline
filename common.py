"""Pieces shared by the node modules."""


class AnyType(str):
    """A socket type that matches every other type (ComfyUI's "*" idiom)."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")
