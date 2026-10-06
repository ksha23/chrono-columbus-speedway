"""Which stock Chrono vehicles become parked-car models, and where Chrono puts their wheels.

Each entry names the files under Chrono's data/vehicle directory and says how the chassis
mesh's materials map to parts. A material with plain colours maps by name. One whose colours
are in a texture gives the texture, a second colour variant of it (the texels that differ
are the paint) and the opacity map (the see-through texels are the glass).
"""
import json
import os
import re

MODELS = [
    {
        "name": "sedan", "what": "four-door sedan",
        "vehicle": "sedan/vehicle/Sedan_Vehicle.json",
        "chassis": "sedan/sedan_chassis_vis.obj", "mtl": "sedan_chassis_vis.mtl",
        # 001 paint, 004 glass, 005 black trim and grille, 006 mirror glass, 007 headlamps,
        # 008 tail lamps. Left out: 002 steering wheel, 003 and 009 to 012 seats.
        "materials": {"Material.001": "body", "Material.004": "glass", "Material.005": "trim",
                      "Material.006": "bright", "Material.007": "bright", "Material.008": "taillights"},
        "rim": "sedan/sedan_rim.obj", "tyre": "sedan/sedan_tire.obj",
        # The body is an open shell with no floor or arch linings.
        "core": True,
        # Chrono's rear suspension sits 0.125 m lower on this chassis than its front one, so
        # the Chrono sedan rides nose down with a hand's width of air over the rear tyres.
        # Parked, the rear wheels go up into their arches like the front ones.
        "spindle_lift": {1: 0.125},
    },
    {
        "name": "hatchback", "what": "low five-door fastback (the Chrono Audi)",
        "vehicle": "audi/json/audi_Vehicle.json",
        "chassis": "audi/audi_chassis.obj",
        "materials": {
            "phong1SG": {"texture": "textures/Audi_Whole_Texturing_031021_Chassis1_BaseColor_Red.jpg",
                         "variant": "textures/Audi_Chassis_BaseColor_white.jpg",
                         "opacity": "textures/Chassis1_Opacity.jpg"},
            "phong2SG": {"texture": "textures/Audi_Whole_Texturing_031021_Interior_Aux_BaseColor.jpg"},
        },
        "also": ["audi/textures/Audi_Whole_Texturing_031021_Chassis1_BaseColor_Red.jpg",
                 "audi/textures/Audi_Chassis_BaseColor_white.jpg", "audi/textures/Chassis1_Opacity.jpg",
                 "audi/textures/Audi_Whole_Texturing_031021_Interior_Aux_BaseColor.jpg",
                 "audi/textures/Audi_Whole_Texturing_031021_Wheels_BaseColor.jpg"],
        "rim": "audi/audi_rim.obj", "tyre": "audi/audi_tire.obj",
    },
    {
        "name": "suv", "what": "large off-road SUV with roof rack, bull bar and spare wheel (the Chrono Nissan Patrol)",
        "vehicle": "Nissan_Patrol/json/suv_Vehicle.json",
        "chassis": "Nissan_Patrol/suv_chassis.obj",
        "materials": {
            "lambert5SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Chassis_BaseColor.jpg",
                           "variant": "textures/Nissan_Chassis_BaseColor_Red.jpg",
                           "opacity": "textures/Nissan_Chassis_Opacity.jpg"},
            "lambert6SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Exterior_BaseColor.jpg"},
            # The spare wheel on the tailgate.
            "lambert7SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Wheel_BaseColor.jpg",
                           "dark": "tyres", "bright": "wheels"},
            "lambert2SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Lights_BaseColor.jpg"},
            "lambert3SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Lights_BaseColor.jpg"},
            "lambert4SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Lights_BaseColor.jpg"},
            # Left out: lambert8SG, the cabin.
        },
        "also": ["Nissan_Patrol/textures/Nissan_Chassis_OBJ_0331_Nissan_Chassis_BaseColor.jpg",
                 "Nissan_Patrol/textures/Nissan_Chassis_BaseColor_Red.jpg", "Nissan_Patrol/textures/Nissan_Chassis_Opacity.jpg",
                 "Nissan_Patrol/textures/Nissan_Chassis_OBJ_0331_Nissan_Exterior_BaseColor.jpg",
                 "Nissan_Patrol/textures/Nissan_Chassis_OBJ_0331_Nissan_Wheel_BaseColor.jpg",
                 "Nissan_Patrol/textures/Nissan_Chassis_OBJ_0331_Nissan_Lights_BaseColor.jpg"],
        "rim": "Nissan_Patrol/suv_rim.obj", "tyre": "Nissan_Patrol/suv_tire.obj",
        # The rim is a flat disc with the spokes painted on it.
        "rim_pattern": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Wheel_BaseColor.jpg", "dark_below": 0.6},
    },
    {
        # Chrono's data has no pickup. This is the SUV above with its body cut down: see carpickup.py.
        "name": "pickup", "what": "crew-cab pickup, made by cutting down the SUV's body",
        "vehicle": "Nissan_Patrol/json/suv_Vehicle.json",
        "chassis": "Nissan_Patrol/suv_chassis.obj",
        "materials": {
            "lambert5SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Chassis_BaseColor.jpg",
                           "variant": "textures/Nissan_Chassis_BaseColor_Red.jpg",
                           "opacity": "textures/Nissan_Chassis_Opacity.jpg"},
            "lambert6SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Exterior_BaseColor.jpg"},
            "lambert2SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Lights_BaseColor.jpg"},
            "lambert3SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Lights_BaseColor.jpg"},
            "lambert4SG": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Lights_BaseColor.jpg"},
            # Left out: lambert8SG, the cabin, and lambert7SG, the spare wheel.
        },
        # The cab ends behind the rear doors, 0.35 m ahead of the rear axle. The cut along the
        # window sills is 1.22 m up, and the roof rack, everything above 1.865 m, goes.
        "pickup": {"cab_back": 0.35, "belt": 1.22, "roof": 1.865},
        "also": ["Nissan_Patrol/textures/Nissan_Chassis_OBJ_0331_Nissan_Chassis_BaseColor.jpg",
                 "Nissan_Patrol/textures/Nissan_Chassis_BaseColor_Red.jpg", "Nissan_Patrol/textures/Nissan_Chassis_Opacity.jpg",
                 "Nissan_Patrol/textures/Nissan_Chassis_OBJ_0331_Nissan_Exterior_BaseColor.jpg",
                 "Nissan_Patrol/textures/Nissan_Chassis_OBJ_0331_Nissan_Wheel_BaseColor.jpg",
                 "Nissan_Patrol/textures/Nissan_Chassis_OBJ_0331_Nissan_Lights_BaseColor.jpg"],
        "rim": "Nissan_Patrol/suv_rim.obj", "tyre": "Nissan_Patrol/suv_tire.obj",
        "rim_pattern": {"texture": "textures/Nissan_Chassis_OBJ_0331_Nissan_Wheel_BaseColor.jpg", "dark_below": 0.6},
    },
    {
        "name": "van", "what": "two-tone microbus (the Chrono VW van)",
        "vehicle": "VW_microbus/json/van_Vehicle.json",
        # The vehicle's JSON names van_chassis.obj, which has no colours. The colour variants
        # are the same body 1.1 times the size and moved: x' = 1.1 x + (0.0584, 0, -0.0021),
        # every vertex of the plain mesh matching one of theirs to within 0.1 mm. The model
        # uses a variant for its colours, brought back to the size the wheels are placed for.
        "chassis": "VW_microbus/van_chassis_cream.obj", "frame": (1.1, (0.0584, 0.0, -0.0021)),
        "materials": {
            "lambert13SG": {"texture": "textures/Volks_Van_Chassis_BaseColor_Cream.jpg",
                            "variant": "textures/Volks_Van_Chassis_BaseColor_Yellow.jpg",
                            "opacity": "textures/Volks_Van_Chassis_Opacity.jpg", "bright": "top"},
            "lambert14SG": {"texture": "textures/Volks_Van_Texturing_Volks_Van_Exteriors_BaseColor.jpg"},
        },
        "also": ["VW_microbus/van_chassis.obj", "VW_microbus/textures/Volks_Van_Chassis_BaseColor_Cream.jpg",
                 "VW_microbus/textures/Volks_Van_Chassis_BaseColor_Yellow.jpg", "VW_microbus/textures/Volks_Van_Chassis_Opacity.jpg",
                 "VW_microbus/textures/Volks_Van_Texturing_Volks_Van_Exteriors_BaseColor.jpg"],
        # These two meshes come without a material file.
        "rim": "VW_microbus/van_rim.obj", "tyre": "VW_microbus/van_tire.obj",
        "rim_colour": (0.6, 0.6, 0.6), "tyre_colour": (0.02, 0.02, 0.02),
    },
]


def read_json(path):
    """A Chrono JSON file, which may carry // comments."""
    with open(path) as f:
        return json.loads(re.sub(r"//[^\n]*", "", f.read()))


def wheel_positions(data_dir, vehicle_json):
    """Spindle centres in the chassis frame, as the vehicle's JSON files give them:
    {(axle, side): (x, y, z)} with side 0 left and 1 right.

    A spindle sits at its suspension's "Suspension Location" in the vehicle file plus the
    "Spindle" "COM" point in the suspension file, mirrored in y for the right side.
    """
    root = os.path.join(data_dir, "vehicle")
    vehicle = read_json(os.path.join(root, vehicle_json))
    out = {}
    for index, axle in enumerate(vehicle["Axles"]):
        at = axle["Suspension Location"]
        spindle = read_json(os.path.join(root, axle["Suspension Input File"]))["Spindle"]["COM"]
        out[(index, 0)] = (at[0] + spindle[0], at[1] + spindle[1], at[2] + spindle[2])
        out[(index, 1)] = (at[0] + spindle[0], at[1] - spindle[1], at[2] + spindle[2])
    return out
