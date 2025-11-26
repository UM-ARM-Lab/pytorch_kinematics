import collections
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import yaml
from lxml import etree


def xml_string(rootXml: etree._Element, addHeader: bool = True) -> str:
    # Meh
    xmlString: str = etree.tostring(rootXml, pretty_print=True, encoding="unicode")
    if addHeader:
        xmlString = '<?xml version="1.0"?>\n' + xmlString
    return xmlString


def dict_sub(obj: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: obj[key] for key in keys}


def node_add(
    doc: etree._Element,
    sub: Optional[Union[str, etree._Element]],
) -> Optional[etree._Element]:
    if sub is None:
        return None
    if isinstance(sub, str):
        return etree.SubElement(doc, sub)
    elif isinstance(sub, etree._Element):
        doc.append(sub)  # This screws up the rest of the tree for prettyprint
        return sub
    else:
        raise Exception("Invalid sub value")


def pfloat(x: Any) -> str:
    return str(x).rstrip(".")


def xml_children(node: etree._Element) -> List[etree._Element]:
    children: List[etree._Element] = node.getchildren()

    def predicate(n: etree._Element) -> bool:
        return not isinstance(n, etree._Comment)

    return list(filter(predicate, children))


def isstring(obj: Any) -> bool:
    try:
        return isinstance(obj, basestring)  # type: ignore[name-defined]
    except NameError:
        return isinstance(obj, str)


def to_yaml(obj: Any) -> Any:
    """Simplify yaml representation for pretty printing"""
    # Is there a better way to do this by adding a representation with
    # yaml.Dumper?
    # Ordered dict: http://pyyaml.org/ticket/29#comment:11
    if obj is None or isstring(obj):
        out: Any = str(obj)
    elif type(obj) in [int, float, bool]:
        return obj
    elif hasattr(obj, "to_yaml"):
        out = obj.to_yaml()
    elif isinstance(obj, etree._Element):
        out = etree.tostring(obj, pretty_print=True)
    elif isinstance(obj, dict):
        out = {}
        for var, value in obj.items():
            out[str(var)] = to_yaml(value)
    elif hasattr(obj, "tolist"):
        # For numpy objects
        out = to_yaml(obj.tolist())
    elif isinstance(obj, collections.abc.Iterable):
        out = [to_yaml(item) for item in obj]
    else:
        out = str(obj)
    return out


class SelectiveReflection:
    def get_refl_vars(self) -> List[str]:
        return list(vars(self).keys())


class YamlReflection(SelectiveReflection):
    def to_yaml(self) -> Any:
        raw = {var: getattr(self, var) for var in self.get_refl_vars()}
        return to_yaml(raw)

    def __str__(self) -> str:
        # Good idea? Will it remove other important things?
        return yaml.dump(self.to_yaml()).rstrip()
