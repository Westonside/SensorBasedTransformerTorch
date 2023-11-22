import json



def load_configuration(configuration_file: json):
    with open(configuration_file) as f:
        configuration = json.load(f)
    print(configuration['configurations'])
    return configuration


def match_config_modal(configuration, key):
    if configuration.get(key) is None:
        print(f"configuration does not have a {key}")
        return None
    else:
        return configuration[key]

modals = {
    "accelerometer": range(0, 3),
    "gyro": range(3, 6),
    "mag": range(6, 9),
    "accelerometer gyro": range(0, 6),
}
