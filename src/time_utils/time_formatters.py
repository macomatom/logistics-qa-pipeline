def format_time(seconds, include_milliseconds=False):
    """Formats time in HH:MM:SS or MM:SS format with optional milliseconds.
    
    Args:
        seconds (float): Time duration in seconds.
        include_milliseconds (bool): Whether to include milliseconds in output.
    
    Returns:
        str: Formatted time string.
    """
    if seconds < 0:
        return "Invalid time (negative value)"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(seconds % 60)
    millisec = int((seconds % 1) * 1000)  # Extract milliseconds
    
    # Build the time string
    if hours > 0:
        time_str = f"{hours}:{minutes:02d}:{sec:02d}"
    else:
        time_str = f"{minutes:02d}:{sec:02d}"
    
    # Append milliseconds if enabled
    if include_milliseconds:
        time_str += f".{millisec:03d}"
    
    return f"{time_str} ({seconds:.2f} sec)"


## Povodna verzia
# def format_time(seconds):
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     sec = int(seconds % 60)
#     if hours > 0:
#         return f"{hours}:{minutes:02d}:{sec:02d} ({seconds:.2f} sec)"
#     else:
#         return f"{minutes:02d}:{sec:02d} ({seconds:.2f} sec)"