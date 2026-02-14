"""
Library to generate Webots worlds.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional
from typing_extensions import override

# =================================================================
#  Base definitions
# =================================================================
class ComponentBase(ABC):
    """Base Webots component"""

    @abstractmethod
    def get_string(self, indent: str = '  ') -> str:
        """Get string representation."""
        raise NotImplementedError()

@dataclass
class ComponentGroup(ComponentBase):
    """Group of components."""
    children: list[ComponentBase] = field(default_factory=list)
    name: Optional[str] = None

    @override
    def get_string(self, indent: str = '  ') -> str:
        """Get string representation."""
        prefix = '' if not self.name else f'DEF {self.name} '
        prefix += "Group {\n" + indent + 'children ' + '[\n'

        # Indent every line of each child's block so children are nested correctly
        child_blocks: list[str] = []
        for child in self.children:
            child_str = child.get_string(indent=indent)
            indented_child = '\n'.join(f'{indent*2}{line}' if line != '' else '' for line in child_str.splitlines())
            child_blocks.append(indented_child)

        children_str = '\n'.join(child_blocks)

        suffix = '\n' + indent + ']\n}'

        return prefix + children_str + suffix

# =================================================================
#  Components
# =================================================================
@dataclass
class Floor(ComponentBase):
    """Floor component."""
    name: str
    size: tuple[float, float]  # width, depth
    translation: tuple[float, float, float] = (0.0, 0.5, -0.1)

    @override
    def get_string(self, indent: str = '  ') -> str:
        content =   'Solid {\n'
        content += f'{indent}translation {self.translation[0]} {self.translation[1]} {self.translation[2]}\n'
        content += f'{indent}children ' + '[\n'
        content += f'{indent*2}Shape ' + '{\n'
        content += f'{indent*3}appearance PBRAppearance ' + '{\n'
        content += f'{indent*3}' + '}\n'
        content += f'{indent*3}geometry Box ' + '{\n'
        content += f'{indent*4}size {self.size[0]} {self.size[1]} 0.1\n'
        content += f'{indent*3}' + '}\n'
        content += f'{indent*2}' + '}\n'
        content += f'{indent}]' + '\n'
        content += f'{indent}name "{self.name}"' + '\n'
        content += f'{indent}boundingObject Shape ' + '{\n'
        content += f'{indent*2}geometry Box ' + '{\n'
        content += f'{indent*3}size {self.size[0]} {self.size[1]} 0.1\n'
        content += f'{indent*2}' + '}\n'
        content += f'{indent}' + '}\n'
        content += '}' + '\n'
        return content
    
@dataclass
class CoVAPSyCar(ComponentBase):
    """CoVAPSy car component."""
    name: str
    controller: Literal['<none>', '<extern>', '<generic>'] | str
    color: tuple[float, float, float]  # RGB 0-1
    proto_name: str = 'TT02_2023b'
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)          # x, y, z in meters
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0) # axis x, y, z and angle in radians
    camera_name: None | str = "camera_gilbert" # None if no camera, otherwise name of the camera device

    @override
    def get_string(self, indent: str = '  ') -> str:
        prefix = self.proto_name + " {\n"
        suffix = '}'

        content =  f'{indent}name "{self.name}"\n'
        content += f'{indent}controller "{self.controller}"\n'
        content += f'{indent}color {self.color[0]} {self.color[1]} {self.color[2]}\n'
        content += f'{indent}translation {self.translation[0]} {self.translation[1]} {self.translation[2]}\n'
        content += f'{indent}rotation {self.rotation[0]} {self.rotation[1]} {self.rotation[2]} {self.rotation[3]}\n'

        # Camera
        if self.camera_name:
            content += f'{indent}children ' + '[\n'
            content += f'{indent*2}Solid ' + '{\n'
            content += f'{indent*3}translation 0.16 0 0.11' + '\n'
            content += f'{indent*3}rotation 0 0 1 0' + '\n'
            content += f'{indent*3}children [' + '\n'
            content += f'{indent*4}Shape ' + '{\n'
            content += f'{indent*5}appearance PBRAppearance ' + '{\n'
            content += f'{indent*6}baseColor 1.0 1.0 1.0' + '\n'
            content += f'{indent*6}metalness 0' + '\n'
            content += f'{indent*5}' + '}\n'
            content += f'{indent*5}geometry Box ' + '{\n'
            content += f'{indent*6}size 0.01 0.05 0.05' + '\n'
            content += f'{indent*5}' + '}\n'
            content += f'{indent*4}' + '}\n'
            content += f'{indent*4}Camera ' + '{\n'
            content += f'{indent*5}name "{self.camera_name}"\n'
            content += f'{indent*5}width 200\n'
            content += f'{indent*5}height 30\n'
            content += f'{indent*4}' + '}\n'
            content += f'{indent*3}' + ']\n'
            content += f'{indent*2}' + '}\n'
            content += f'{indent}]' + '\n'
        
        return prefix + content + suffix

@dataclass
class BorderStraight(ComponentBase):
    """Straight border track component."""
    name: str
    length: float  # in meters
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)          # x, y, z in meters
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0) # axis x, y, z and angle in radians
    color: tuple[float, float, float] = (1.0, 0.0, 0.0)  # RGB 0-1

    @override
    def get_string(self, indent: str = '  ') -> str:
        content =   'Solid {\n'
        content += f'{indent}translation {self.translation[0]} {self.translation[1]} {self.translation[2]}\n'
        content += f'{indent}rotation {self.rotation[0]} {self.rotation[1]} {self.rotation[2]} {self.rotation[3]}\n'
        content += f'{indent}children ' + '[\n'
        content += f'{indent*2}Shape ' + '{\n'
        content += f'{indent*3}appearance PBRAppearance ' + '{\n'
        content += f'{indent*4}baseColor {self.color[0]} {self.color[1]} {self.color[2]}\n'
        content += f'{indent*4}metalness 0\n'
        content += f'{indent*3}' + '}\n'
        content += f'{indent*3}geometry Box ' + '{\n'
        content += f'{indent*4}size {self.length} 0.1 0.2\n'
        content += f'{indent*3}' + '}\n'
        content += f'{indent*2}' + '}\n'
        content += f'{indent}]' + '\n'
        content += f'{indent}name "{self.name}"' + '\n'
        content += f'{indent}boundingObject Box ' + '{\n'
        content += f'{indent*2}size {self.length} 0.1 0.2\n'
        content += f'{indent}' + '}\n'
        content += '}' + '\n'
        
        return content

@dataclass
class BorderCurve(ComponentBase):
    """Curved border track component."""
    name: str
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0)
    color: tuple[float, float, float] = (1.0, 0.0, 0.0)

    @override
    def get_string(self, indent: str = '  ') -> str:
        content =  'Solid {\n'
        content += f'{indent}translation {self.translation[0]} {self.translation[1]} {self.translation[2]}\n'
        content += f'{indent}rotation {self.rotation[0]} {self.rotation[1]} {self.rotation[2]} {self.rotation[3]}\n'
        content += f'{indent}children ' + '[\n'
        content += f'{indent*2}Shape ' + '{\n'
        content += f'{indent*3}appearance Appearance ' + '{\n'
        content += f'{indent*4}material Material ' + '{\n'
        content += f'{indent*5}ambientIntensity 0.5\n'
        content += f'{indent*5}diffuseColor {self.color[0]} {self.color[1]} {self.color[2]}\n'
        content += f'{indent*5}emissiveColor {self.color[0]} {self.color[1]} {self.color[2]}\n'
        content += f'{indent*5}specularColor {self.color[0]} {self.color[1]} {self.color[2]}\n'
        content += f'{indent*4}' + '}\n'
        content += f'{indent*3}' + '}\n'
        content += f'{indent*3}geometry Mesh ' + '{\n'
        content += f'{indent*4}url ' + '[\n'
        content += f'{indent*5}"ImageToStl_virage.obj"\n'
        content += f'{indent*4}' + ']\n'
        content += f'{indent*3}' + '}\n'
        content += f'{indent*2}' + '}\n'
        content += f'{indent}]' + '\n'
        content += f'{indent}name "{self.name}"' + '\n'
        content += f'{indent}boundingObject Mesh ' + '{\n'
        content += f'{indent*2}url ' + '[\n'
        content += f'{indent*3}"ImageToStl_virage.obj"\n'
        content += f'{indent*2}' + ']\n'
        content += f'{indent}' + '}\n'
        content += '}' + '\n'
        return content
    
# =================================================================
#  World generator
# =================================================================
@dataclass
class World(ComponentBase):
    """Webots world generator class."""
    world_path: str
    components: list[ComponentBase] = field(default_factory=list)
    viewpoint_arg: str = '''
orientation -0.49747884224006406 0.6050877728050384 0.6215976099739475 2.081904069517053
position 6.158656608120317 -5.250076751623688 46.317718817228396
followType "Mounted Shot"'''

    @override
    def get_string(self, indent: str = '  ') -> str:
        """Get string representation."""
        prefix = ('''#VRML_SIM R2025a utf8

        EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2022b/projects/objects/backgrounds/protos/TexturedBackground.proto"
        EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2022b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
        EXTERNPROTO "../protos/TT02_2023b.proto"

        WorldInfo {
        }
        Viewpoint {
        ''' + self.viewpoint_arg.lstrip('\n') +
        '''
        }
        TexturedBackground {
        }
        TexturedBackgroundLight {
        }
        ''').replace('        ', '')
        suffix = ''

        components_str = '\n'.join(component.get_string(indent=indent) for component in self.components)

        return prefix + components_str + '\n' + suffix
