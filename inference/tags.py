ANIMAL = {"bird", "cat", "deer", "dog", "frog", "horse"}
VEHICLE = {"airplane", "automobile", "ship", "truck"}

def business_tag(raw_class):
    if raw_class in ANIMAL:
        return "animal"
    if raw_class in VEHICLE:
        return "vehicle"
    return "other"
