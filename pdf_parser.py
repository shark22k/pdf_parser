# Databricks notebook source
def filter_words(words_data, words_to_filter):
    # filter words from a list of words
    # words_data: a list of words extracted from a pdf
    # words_to_filter: a list of words to be filtered, i.e. removed from the list
    filtered_words = [word for word in words_data if not word['text'] in words_to_filter]
    return filtered_words


def build_feature_pool(filtered_words):
    # build a feature pool from a list of words
    # filtered_words: a list of words to be used to build the feature pool
    feature_pool = [{
        'text': word['text'],
        'x0': word['x0'],
        'x1': word['x1'],
        'top': word['top'],
        'bottom': word['bottom'],
        'doctop': word['doctop'],
        'center_x': (word['x0'] + word['x1']) / 2,
        'center_y': (word['top'] + word['bottom']) / 2
    } for word in filtered_words]
    return feature_pool


def get_lines(orientation, distance_threshold, feature_pool, region):
    """
    Clusters words based on the specified orientation and draws lines on a plot.

    Parameters:
    - orientation: 'horizontal' or 'vertical' to specify the line orientation.
    - distance_threshold: The clustering distance threshold.
    - feature_pool: List of dictionaries, each containing word features and text.
    - region : a dictionary representing the bottom-left and top-right corners of the region
    """
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    from collections import defaultdict

    # Select features based on orientation
    if orientation == 'horizontal':
        features = np.array([[word['center_y'], word['doctop']] for word in feature_pool])
    elif orientation == 'vertical':
        features = np.array([[word['center_x'], word['x0']] for word in feature_pool])
    else:
        raise ValueError("Orientation must be 'horizontal' or 'vertical'")

    # Perform Agglomerative Clustering
    hierarchical_clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='ward',
        metric='euclidean'
    )
    hierarchical_clustering.fit(features)

    # Extract cluster labels
    cluster_labels = hierarchical_clustering.labels_

    # Creating a dictionary to hold text labels for each cluster including bounding box dimensions associated with
    # each word.
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        word_info = feature_pool[i]
        clusters[label].append({
            'text': word_info['text'],
            'bounding_box': {
                'x0': word_info['x0'],
                'x1': word_info['x1'],
                'top': word_info['top'],
                'bottom': word_info['bottom'],
                'doctop': word_info['doctop']
            }
        })

    # Calculate the top or left position for each cluster, according too the orientation
    for cluster, words in clusters.items():
        if orientation == 'horizontal':
            cluster_value = sorted((list(word['bounding_box']['top'] for word in words)))[0]
        elif orientation == 'vertical':
            cluster_value = sorted((list(word['bounding_box']['x0'] for word in words)))[0]
        clusters[cluster] = {
            'cluster_value': cluster_value
        }

    cluster_values = [cluster['cluster_value'] for _, cluster in clusters.items()]

    if orientation == 'horizontal':
        cluster_values.append(region['bottom'])
        return [[[region['x0'], x], [region['x1'], x]] for x in sorted(cluster_values)]
    elif orientation == 'vertical':
        cluster_values.append(region['x1'])
        return [[[x, region['top']], [x, region['bottom']]] for x in sorted(cluster_values)]


# for removing rows and/or columns
def remove_by_indices(original_list, indices_to_remove):
    return [item for idx, item in enumerate(original_list) if idx not in indices_to_remove]


# for limiting column to a row
def limit_columns(vertical_lines, horizontal_lines, column_row_indices):
    def limit_column(column_index, row_index):
        # Ensure the column_index is within the bounds of the list
        if column_index < 0 or column_index >= len(vertical_lines):
            raise IndexError("column_index out of range")
        top_left_vertical, bottom_right_vertical = vertical_lines[column_index]
        top_left_horizontal, bottom_right_horizontal = horizontal_lines[row_index]
        top_left_vertical[1] = top_left_horizontal[1]
        vertical_lines[column_index][0] = top_left_vertical

    for column, row in column_row_indices:
        limit_column(column, row)
    return vertical_lines


def move_columns(vertical_lines, indices_positions):
    def move_column(index, movPos):
        vertical_lines[index][0][0] += movPos
        vertical_lines[index][1][0] += movPos

    for index, position in indices_positions:
        move_column(index, position)
    return vertical_lines


def get_vertical_horizontal_plotting_lines(pdf_plumber_object, page_number, region, adjustments=None,
                                           horizontal_threshold=10,
                                           vertical_threshold=150,
                                           lines_orientation_type='both',
                                           header_region=None,
                                           footer_region=None):
    page = pdf_plumber_object.pages[page_number - 1]
    crop_bbox = (region['x0'], region['top'], region['x1'], region['bottom'])
    cropped_page = page.within_bbox(crop_bbox)
    words_list = cropped_page.extract_words()
    filtered_words = filter_words(words_list, ['.', ','])
    features = build_feature_pool(filtered_words)

    if header_region is not None:
        header_lines = [
            (header_region['x0'], header_region['top'], header_region['x1'], header_region['top']),
            # (header_region['x0'], header_region['top'], header_region['x0'], header_region['bottom']),
            # (header_region['x0'], header_region['bottom'], header_region['x1'], header_region['bottom']),
            (header_region['x1'], header_region['bottom'], header_region['x0'], header_region['bottom'])
        ]
    else:
        header_lines = []

    if footer_region is not None:
        footer_lines = [
            (footer_region['x0'], footer_region['top'], footer_region['x1'], footer_region['top']),
            # (footer_region['x0'], footer_region['top'], footer_region['x0'], footer_region['bottom']),
            # (footer_region['x0'], footer_region['bottom'], footer_region['x1'], footer_region['bottom']),
            (footer_region['x1'], footer_region['bottom'], footer_region['x0'], footer_region['bottom'])
        ]
    else:
        footer_lines = []

    if lines_orientation_type == 'both':
        horizontal_lines = get_lines('horizontal', horizontal_threshold, features, region)
        vertical_lines = get_lines('vertical', vertical_threshold, features, region)
        # section for adjustments
        if adjustments:
            if adjustments['rows_remove_indices']:
                horizontal_lines = remove_by_indices(horizontal_lines, adjustments['rows_remove_indices'])
            if adjustments['columns_remove_indices']:
                vertical_lines = remove_by_indices(vertical_lines, adjustments['columns_remove_indices'])
            if adjustments['column_row_indices_limit_columns']:
                vertical_lines = limit_columns(vertical_lines, horizontal_lines,
                                               adjustments['column_row_indices_limit_columns'])
            if adjustments['column_position_move_coulmns']:
                vertical_lines = move_columns(vertical_lines, adjustments['column_position_move_coulmns'])
        flatten_horizontal_lines = [(line[0][0], line[0][1], line[1][0], line[1][1]) for line in horizontal_lines]
        flatten_vertical_lines = [(line[0][0], line[0][1], line[1][0], line[1][1]) for line in vertical_lines]
        return flatten_horizontal_lines + flatten_vertical_lines + header_lines + footer_lines

    elif lines_orientation_type == 'horizontal':
        horizontal_lines = get_lines('horizontal', horizontal_threshold, features, region)
        if adjustments:
            if adjustments['rows_remove_indices']:
                horizontal_lines = remove_by_indices(horizontal_lines, adjustments['rows_remove_indices'])
        return [(line[0][0], line[0][1], line[1][0], line[1][1]) for line in horizontal_lines] + header_lines + footer_lines

    elif lines_orientation_type == 'vertical':
        vertical_lines = get_lines('vertical', vertical_threshold, features, region)
        if adjustments:
            if adjustments['columns_remove_indices']:
                vertical_lines = remove_by_indices(vertical_lines, adjustments['columns_remove_indices'])
            if adjustments['column_position_move_coulmns']:
                vertical_lines = move_columns(vertical_lines, adjustments['column_position_move_coulmns'])
        return [(line[0][0], line[0][1], line[1][0], line[1][1]) for line in vertical_lines] + header_lines + footer_lines


def adjust_for_orientation_ccw(line, page):
    """
    Adjust the line coordinates for a 90-degree counterclockwise rotation based on the page's dimensions.

    Args:
    - line (tuple): Original line coordinates as (start_x, start_y, end_x, end_y).
    - page (fitz.Page): The page object from PyMuPDF (fitz).

    Returns:
    - tuple: Adjusted start and end positions as ((new_start_x, new_start_y), (new_end_x, new_end_y)).
    """
    width, height = page.rect.width, page.rect.height  # Get the page dimensions

    # Extract original line coordinates
    start_x, start_y, end_x, end_y = line

    # Adjust coordinates for 90-degree counterclockwise rotation
    new_start_x = start_y
    new_start_y = width - start_x
    new_end_x = end_y
    new_end_y = width - end_x

    # Return the adjusted start and end positions
    return (new_start_x, new_start_y), (new_end_x, new_end_y)


def adjust_for_orientation_90_cw(line, page):
    """
    Adjust the line coordinates for a 90-degree clockwise rotation based on the page's dimensions.

    Args:
    - line (tuple): Original line coordinates as (start_x, start_y, end_x, end_y).
    - page (fitz.Page): The page object from PyMuPDF (fitz).

    Returns:
    - tuple: Adjusted start and end positions as ((new_start_x, new_start_y), (new_end_x, new_end_y)).
    """
    width, height = page.rect.width, page.rect.height  # Get the page dimensions

    # Extract original line coordinates
    start_x, start_y, end_x, end_y = line

    # Adjust coordinates for 90-degree clockwise rotation
    new_start_x = height - start_y
    new_start_y = start_x
    new_end_x = height - end_y
    new_end_y = end_x

    # Return the adjusted start and end positions
    return (new_start_x, new_start_y), (new_end_x, new_end_y)


def draw_lines_on_pdf(pdf_blob, demarcation_settings):
    import pdfplumber
    import fitz  # PyMuPDF
    from io import BytesIO
    # Step 1: Open the PDF blob using pdfplumber
    with pdfplumber.open(BytesIO(pdf_blob)) as pdf:
        # Step 3: Iterate through each page in the PDF file
        page_demarcation_info = {}
        for page_number, page_info in demarcation_settings.items():
            page_data = {'rotation': None}
            lines_on_a_page = []
            for page_info_key, page_info_value in page_info.items():
                if page_info_key == 'rotation':
                    page_data['rotation'] = page_info_value
                    continue
                region = page_info_value['region']
                adjustments = page_info_value['adjustments']
                vertical_threshold = page_info_value.get('vertical_threshold', 150)
                lines_orientation_type = page_info_value.get('lines_orientation', 'both')
                header_region = page_info_value.get('header_region', None)
                footer_region = page_info_value.get('footer_region', None)
                lines = get_vertical_horizontal_plotting_lines(
                    pdf,
                    int(page_number),
                    region,
                    adjustments,
                    vertical_threshold=vertical_threshold,
                    lines_orientation_type=lines_orientation_type,
                    header_region=header_region,
                    footer_region=footer_region
                )
                lines_on_a_page.extend(lines)
            page_data['lines'] = lines_on_a_page
            page_demarcation_info[int(page_number) - 1] = page_data

    # Step 4: Open pdf blob with fitz (PyMuPDF) to draw lines
    doc = fitz.open("pdf", pdf_blob)  # Reopen the PDF blob with fitz for drawing
    for page_number, page_lines_orientation_data in page_demarcation_info.items():
        page = doc[page_number]
        shape = page.new_shape()
        for line in page_lines_orientation_data['lines']:
            if page_lines_orientation_data['rotation']:
                rotation_function = page_lines_orientation_data['rotation']
                start_pos, end_pos = rotation_function(line, page)
            else:
                start_pos, end_pos = (line[0], line[1]), (line[2], line[3])
            shape.draw_line(start_pos, end_pos)
        # Finalize drawing
        shape.finish(color=(1, 0, 0), width=1.5, fill=(1, 0, 0))  # Example stroke color red, and line width
        shape.commit()

    # Return the modified PDF as bytes
    return doc.tobytes()


def extract_table_from_region(pdf_blob, page_number, table_region=None, header_region=None, footer_region=None):
    """
    Extracts a table from a specified region on a specific page of a PDF.

    Args:
    - pdf_name (str): The path to the PDF file.
    - page_number (int): The page number from which to extract the table (0-based index).
    - region (dict): A dictionary specifying the cropping region with keys 'x0', 'y0', 'x1', 'y1'.
                     These correspond to the coordinates of the bottom-left (x0, y0) and
                     top-right (x1, y1) corners of the desired region.

    Returns:
    - table (list of lists): The extracted table as a list of rows, with each row being a list of cell contents.
    """
    import pdfplumber
    from io import BytesIO

    with pdfplumber.open(BytesIO(pdf_blob)) as pdf:
        # Ensure the page number is within the valid range
        page_number -= 1
        if page_number < 0 or page_number >= len(pdf.pages):
            print(f"Page number {page_number} is out of range for this PDF.")
            return None

        page = pdf.pages[page_number]
        # Crop the page to the specified region before extracting the table
        # Note: The 'region' argument is expected to be a dictionary with 'x0', 'y0', 'x1', 'y1' keys
        if table_region is not None:
            cropped_page_for_table = page.within_bbox(
                (table_region['x0'], table_region['top'], table_region['x1'], table_region['bottom']))
            # Extract the table from the cropped page
            table = cropped_page_for_table.extract_table()
        else:
            table = page.extract_table()

        # Extracting header of a table if specified
        if header_region is not None:
            cropped_page_for_header = page.within_bbox(
                (header_region['x0'], header_region['top'], header_region['x1'], header_region['bottom']))
            header_text = cropped_page_for_header.extract_text()
        else:
            header_text = ''

        # Extracting footer of a table if specified
        if footer_region is not None:
            cropped_page_for_footer = page.within_bbox(
                (footer_region['x0'], footer_region['top'], footer_region['x1'], footer_region['bottom']))
            footer_text = cropped_page_for_footer.extract_text()
        else:
            footer_text = ''

        return [header_text, footer_text, table]


def clean_and_unique(lst):
    import re
    """Remove None or empty strings, replace newlines and spaces with underscores, and keep unique values while preserving order."""
    seen = set()
    # Replace each instance of \n or whitespace with _
    cleaned_list = [re.sub(r'[\n\s]+', '_', x) for x in lst if x not in [None, ""]]
    return [x for x in cleaned_list if not (x in seen or seen.add(x))]


def create_combinations(hierarchy_lists, prefix='', sep='_'):
    """Recursively build combinations from hierarchical lists."""
    if not hierarchy_lists:
        return [prefix.rstrip(sep)]  # Remove trailing separator
    # Generate combinations for the current hierarchy and proceed recursively
    return [combo for item in hierarchy_lists[0]
            for combo in create_combinations(hierarchy_lists[1:], prefix + item + sep, sep)]


def create_column_hierarchy_ordered(hierarchies):
    """
    Creates structured hierarchical labels from multiple levels of hierarchies,
    maintaining the order of upper hierarchies in the result list for n levels.
    Each hierarchy is formatted as "hierarchy1-hierarchy2-hierarchy3-...",
    maintaining the order and ensuring unique combinations.
    """
    # Clean each hierarchy list to remove None or empty strings, replace \n or \s with _, and ensure uniqueness
    cleaned_hierarchies = [clean_and_unique(hierarchy) for hierarchy in hierarchies.values()]

    # Use a recursive function to combine the cleaned hierarchies into structured labels
    hierarchical_labels = create_combinations(cleaned_hierarchies)

    return hierarchical_labels


def filter_empty_or_none_rows(table):
    return [row for row in table if not all(item in ["", None] for item in row)]


# converting to dataframes

def to_dataframe(columns, rows):
    # Create DataFrame from columns and rows
    import pandas as pd
    return pd.DataFrame(rows, columns=columns)
