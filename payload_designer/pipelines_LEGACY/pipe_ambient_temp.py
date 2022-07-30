"""Calculate the net operating temperature across all components."""
# project
from payload_designer.components.components import Components


def get_components():
    obj = Components()
    return [
        a for a in dir(obj) if not a.startswith("__") and not callable(getattr(obj, a))
    ]


def filter_components(components):
    return [a for a in components if not a.min_operating_temp]


def calculate_ambient_operating_temp():
    all_components = get_components()
    filtered_components = filter_components(all_components)

    operating_ranges = []
    # operating_ranges = [a.calculate_operating_temp_range() for a in filtered_components]

    for comp in filtered_components:
        operating_ranges.append(comp.calculate_operating_temp_range())

    min_range = min(operating_ranges)
    min_component = filtered_components[operating_ranges.index(min_range)]

    print(f"Component: {min_component}")
    print(
        f"range: {min_component.min_operating_temp} - \
        {min_component.max_operating_temp}"
    )

    return (
        min_component,
        min_range,
        min_component.min_operating_temp,
        min_component.max_operating_temp,
    )
