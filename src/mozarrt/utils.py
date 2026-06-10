from typing import Any


def hex_rgb_to_rgba(color_hex: str) -> str:
    normalized = color_hex.lstrip("#")
    return (
        f"{int(normalized[0:2], 16)}-"
        f"{int(normalized[2:4], 16)}-"
        f"{int(normalized[4:6], 16)}-255"
    )


def update_channel_display_metadata(
    *,
    channel_name: str,
    channel_colors: dict[str, str],
    channel_contrast_limits: dict[str, list[float]],
    container_channels_meta: Any,
    image_channels_meta: Any,
) -> None:
    channel_index = image_channels_meta.get_channel_idx(channel_name)
    if channel_name not in channel_colors:
        channel_colors[channel_name] = hex_rgb_to_rgba(
            container_channels_meta.channels[channel_index].channel_visualisation.color
        )

    channel_visualisation = image_channels_meta.channels[
        channel_index
    ].channel_visualisation
    channel_contrast_limits[channel_name][0] = min(
        channel_contrast_limits[channel_name][0], channel_visualisation.start
    )
    channel_contrast_limits[channel_name][1] = max(
        channel_contrast_limits[channel_name][1], channel_visualisation.end
    )
