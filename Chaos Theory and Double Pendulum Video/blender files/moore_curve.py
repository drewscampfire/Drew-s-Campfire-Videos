import bpy
import bmesh
import math

def moore_curve(order):
    def l_system(axiom, rules, iterations):
        for _ in range(iterations):
            axiom = ''.join(rules.get(c, c) for c in axiom)
        return axiom

    def interpret(commands, angle):
        x, y = 0, 0
        direction = 0
        path = [(x, y)]

        for cmd in commands:
            if cmd == 'F':
                x += math.cos(direction)
                y += math.sin(direction)
                path.append((x, y))
            elif cmd == '+':
                direction += angle
            elif cmd == '-':
                direction -= angle

        return path

    axiom = "LFL+F+LFL"
    rules = {
        "L": "-RF+LFL+FR-",
        "R": "+LF-RFR-FL+"
    }

    curve = l_system(axiom, rules, order)
    path = interpret(curve, math.pi / 2)

    return path

def create_curve_object(name, points):
    curve_data = bpy.data.curves.new(name=name, type='CURVE')
    curve_data.dimensions = '2D'

    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(points) - 1)
    for i, point in enumerate(points):
        polyline.points[i].co = (point[0], point[1], 0, 1)

    curve_object = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_object)

    return curve_object

# Generate Moore Curve
order = 3  # You can change this to increase or decrease the complexity
points = moore_curve(order)

# Create Blender curve object
curve_object = create_curve_object("Moore Curve", points)

# Center the curve
curve_object.select_set(True)
bpy.context.view_layer.objects.active = curve_object
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
bpy.ops.object.location_clear()

print("Moore Curve created and added to the scene.")