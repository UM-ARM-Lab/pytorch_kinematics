from .xml_reflection.basics import *
from . import xml_reflection as xmlr

# What is the scope of plugins? Model, World, Sensor?

xmlr.start_namespace('sdf')

name_attribute = xmlr.Attribute('name', str, False)
pose_element = xmlr.Element('pose', 'vector6', False)


class Inertia(xmlr.Object):
    KEYS = ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']

    def __init__(self, ixx=0.0, ixy=0.0, ixz=0.0, iyy=0.0, iyz=0.0, izz=0.0):
        self.ixx = ixx
        self.ixy = ixy
        self.ixz = ixz
        self.iyy = iyy
        self.iyz = iyz
        self.izz = izz

    def to_matrix(self):
        return [
            [self.ixx, self.ixy, self.ixz],
            [self.ixy, self.iyy, self.iyz],
            [self.ixz, self.iyz, self.izz]]


xmlr.reflect(Inertia,
             params=[xmlr.Element(key, float) for key in Inertia.KEYS])


# Pretty much copy-paste... Better method?
# Use multiple inheritance to separate the objects out so they are unique?


class Inertial(xmlr.Object):
    def __init__(self, mass=0.0, inertia=None, pose=None):
        self.mass = mass
        self.inertia = inertia
        self.pose = pose


xmlr.reflect(Inertial, params=[
    xmlr.Element('mass', float),
    xmlr.Element('inertia', Inertia),
    pose_element
])


class Box(xmlr.Object):
    def __init__(self, size=None):
        self.size = size


xmlr.reflect(Box, tag='box', params=[
    xmlr.Element('size', 'vector3')
])


class Cylinder(xmlr.Object):
    def __init__(self, radius=0.0, length=0.0):
        self.radius = radius
        self.length = length


xmlr.reflect(Cylinder, tag='cylinder', params=[
    xmlr.Element('radius', float),
    xmlr.Element('length', float)
])


class Sphere(xmlr.Object):
    def __init__(self, radius=0.0):
        self.radius = radius


xmlr.reflect(Sphere, tag='sphere', params=[
    xmlr.Element('radius', float)
])


class Mesh(xmlr.Object):
    def __init__(self, filename=None, scale=None):
        self.filename = filename
        self.scale = scale


xmlr.reflect(Mesh, tag='mesh', params=[
    xmlr.Element('filename', str),
    xmlr.Element('scale', 'vector3', required=False)
])


class GeometricType(xmlr.ValueType):
    def __init__(self):
        self.factory = xmlr.FactoryType('geometric', {
            'box': Box,
            'cylinder': Cylinder,
            'sphere': Sphere,
            'mesh': Mesh
        })

    def from_xml(self, node, path):
        children = xml_children(node)
        assert len(children) == 1, 'One element only for geometric'
        return self.factory.from_xml(children[0], path=path)

    def write_xml(self, node, obj):
        name = self.factory.get_name(obj)
        child = node_add(node, name)
        obj.write_xml(child)


xmlr.add_type('geometric', GeometricType())


class Script(xmlr.Object):
    def __init__(self, uri=None, name=None):
        self.uri = uri
        self.name = name


xmlr.reflect(Script, tag='script', params=[
    xmlr.Element('name', str, False),
    xmlr.Element('uri', str, False)
])


class Material(xmlr.Object):
    def __init__(self, name=None, script=None):
        self.name = name
        self.script = script


xmlr.reflect(Material, tag='material', params=[
    name_attribute,
    xmlr.Element('script', Script, False)
])


class Visual(xmlr.Object):
    def __init__(self, name=None, geometry=None, pose=None):
        self.name = name
        self.geometry = geometry
        self.pose = pose


xmlr.reflect(Visual, tag='visual', params=[
    name_attribute,
    xmlr.Element('geometry', 'geometric'),
    xmlr.Element('material', Material, False),
    pose_element
])


class Collision(xmlr.Object):
    def __init__(self, name=None, geometry=None, pose=None):
        self.name = name
        self.geometry = geometry
        self.pose = pose


xmlr.reflect(Collision, tag='collision', params=[
    name_attribute,
    xmlr.Element('geometry', 'geometric'),
    pose_element
])


class Dynamics(xmlr.Object):
    def __init__(self, damping=None, friction=None):
        self.damping = damping
        self.friction = friction


xmlr.reflect(Dynamics, tag='dynamics', params=[
    xmlr.Element('damping', float, False),
    xmlr.Element('friction', float, False)
])


class Limit(xmlr.Object):
    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper


xmlr.reflect(Limit, tag='limit', params=[
    xmlr.Element('lower', float, False),
    xmlr.Element('upper', float, False)
])


class Axis(xmlr.Object):
    def __init__(self, xyz=None, limit=None, dynamics=None,
                 use_parent_model_frame=None):
        self.xyz = xyz
        self.limit = limit
        self.dynamics = dynamics
        self.use_parent_model_frame = use_parent_model_frame


xmlr.reflect(Axis, tag='axis', params=[
    xmlr.Element('xyz', 'vector3'),
    xmlr.Element('limit', Limit, False),
    xmlr.Element('dynamics', Dynamics, False),
    xmlr.Element('use_parent_model_frame', bool, False)
])


class Joint(xmlr.Object):
    TYPES = ['unknown', 'revolute', 'gearbox', 'revolute2',
             'prismatic', 'ball', 'screw', 'universal', 'fixed']

    def __init__(self, name=None, parent=None, child=None, joint_type=None,
                 axis=None, pose=None):
        self.aggregate_init()
        self.name = name
        self.parent = parent
        self.child = child
        self.type = joint_type
        self.axis = axis
        self.pose = pose

    # Aliases
    @property
    def joint_type(self): return self.type

    @joint_type.setter
    def joint_type(self, value): self.type = value


xmlr.reflect(Joint, tag='joint', params=[
    name_attribute,
    xmlr.Attribute('type', str, False),
    xmlr.Element('axis', Axis),
    xmlr.Element('parent', str),
    xmlr.Element('child', str),
    pose_element
])


class Link(xmlr.Object):
    def __init__(self, name=None, pose=None, inertial=None, kinematic=False):
        self.aggregate_init()
        self.name = name
        self.pose = pose
        self.inertial = inertial
        self.kinematic = kinematic
        self.visuals = []
        self.collisions = []


xmlr.reflect(Link, tag='link', params=[
    name_attribute,
    xmlr.Element('inertial', Inertial),
    xmlr.Attribute('kinematic', bool, False),
    xmlr.AggregateElement('visual', Visual, var='visuals'),
    xmlr.AggregateElement('collision', Collision, var='collisions'),
    pose_element
])


class Model(xmlr.Object):
    def __init__(self, name=None, pose=None):
        self.aggregate_init()
        self.name = name
        self.pose = pose
        self.links = []
        self.joints = []
        self.joint_map = {}
        self.link_map = {}

        self.parent_map = {}
        self.child_map = {}

    def add_aggregate(self, typeName, elem):
        xmlr.Object.add_aggregate(self, typeName, elem)

        if typeName == 'joint':
            joint = elem
            self.joint_map[joint.name] = joint
            self.parent_map[joint.child] = (joint.name, joint.parent)
            if joint.parent in self.child_map:
                self.child_map[joint.parent].append((joint.name, joint.child))
            else:
                self.child_map[joint.parent] = [(joint.name, joint.child)]
        elif typeName == 'link':
            link = elem
            self.link_map[link.name] = link

    def add_link(self, link):
        self.add_aggregate('link', link)

    def add_joint(self, joint):
        self.add_aggregate('joint', joint)


xmlr.reflect(Model, tag='model', params=[
    name_attribute,
    xmlr.AggregateElement('link', Link, var='links'),
    xmlr.AggregateElement('joint', Joint, var='joints'),
    pose_element
])


class SDF(xmlr.Object):
    def __init__(self, version=None):
        self.version = version


xmlr.reflect(SDF, tag='sdf', params=[
    xmlr.Attribute('version', str, False),
    xmlr.Element('model', Model, False),
])

xmlr.end_namespace()
