import json



def load_configuration(configuration_file: json):
    with open(configuration_file) as f:
        configuration = json.load(f)
    print(configuration['configurations'])
    return configuration


def match_config_key(configuration, key):
    if configuration.get(key) is None:
        print(f"configuration does not have a {key}")
        return None
    else:
        return " ".join(configuration[key])

modals = {
    "accelerometer": range(0, 3),
    "gyroscope": range(3, 6),
    "mag": range(6, 9),
    "accelerometer gyroscope": range(0, 6),
}
