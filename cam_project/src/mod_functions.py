import collections
import six
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


# Helper Code
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Drawing Boxes
def draw_boxes(
    image,
    boxes,
    classes,
    scores,
    category_index,
    object_list,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    mask_alpha=.4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):

	box_to_display_str_map = collections.defaultdict(list)
	box_to_color_map = collections.defaultdict(str)
	box_to_instance_masks_map = {}
	box_to_instance_boundaries_map = {}
	box_to_keypoints_map = collections.defaultdict(list)
	box_to_keypoint_scores_map = collections.defaultdict(list)
	box_to_track_ids_map = {}
	
	classes_list = []
	scores_list = []
	boxes_list = []
	

	if not max_boxes_to_draw:
		max_boxes_to_draw = boxes.shape[0]
		
	for i in range(boxes.shape[0]):
		if max_boxes_to_draw == len(box_to_color_map):
			break
		if scores is None or scores[i] > min_score_thresh:
			box = tuple(boxes[i].tolist())
			if instance_masks is not None:
				box_to_instance_masks_map[box] = instance_masks[i]
			if instance_boundaries is not None:
				box_to_instance_boundaries_map[box] = instance_boundaries[i]
			if keypoints is not None:
				box_to_keypoints_map[box].extend(keypoints[i])
			if keypoint_scores is not None:
				box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
			if track_ids is not None:
				box_to_track_ids_map[box] = track_ids[i]
			if scores is None:
				box_to_color_map[box] = groundtruth_box_visualization_color
			else:
				flag = 0
				display_str = ''
				if not skip_labels: #####
					if not agnostic_mode:
						if classes[i] in six.viewkeys(category_index): ####### Class Found
							class_name = category_index[classes[i]]['name'] 
							if (class_name in object_list):
								classes_list.append(class_name)
								flag = 1
						else:
							class_name = 'N/A'
						display_str = str(class_name)
				if not skip_scores: #####
					if not display_str:
						display_str = '{}%'.format(round(100*scores[i]))
					else: ####### Score Found for the Class
						if(flag == 1):
							display_str = '{}: {}%'.format(display_str, round(100*scores[i])) 
							scores_list.append(scores[i])

				if not skip_track_ids and track_ids is not None:
					if not display_str:
						display_str = 'ID {}'.format(track_ids[i])
					else:
						display_str = '{}: ID {}'.format(display_str, track_ids[i]) 

				box_to_display_str_map[box].append(display_str)
				if agnostic_mode:
					box_to_color_map[box] = 'DarkOrange'
				elif track_ids is not None:
					prime_multipler = _get_multiplier_for_color_randomness()
					box_to_color_map[box] = STANDARD_COLORS[(prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
				else: ####### Boxes to Draw 
					if (flag == 1):
						box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)] 
	
				if (flag == 1): 
					ymin,xmin,ymax,xmax = box
					boxes_list.append(box)
	
	for box, color in box_to_color_map.items():
		ymin, xmin, ymax, xmax = box
		
		if instance_masks is not None:
			vis_util.draw_mask_on_image_array(
				image,
				box_to_instance_masks_map[box],
				color=color,
				alpha=mask_alpha
			)
		if instance_boundaries is not None:
			vis_util.draw_mask_on_image_array(
				image,
				box_to_instance_boundaries_map[box],
				color='red',
				alpha=1.0
			)
		vis_util.draw_bounding_box_on_image_array(
			image,
			ymin,
			xmin,
			ymax,
			xmax,
			color=color,
			thickness=0 if skip_boxes else line_thickness,
			display_str_list=box_to_display_str_map[box],
			use_normalized_coordinates=use_normalized_coordinates)
		if keypoints is not None:
			keypoint_scores_for_box = None
			if box_to_keypoint_scores_map:
				keypoint_scores_for_box = box_to_keypoint_scores_map[box]
			vis_util.draw_keypoints_on_image_array(
				image,
				box_to_keypoints_map[box],
				keypoint_scores_for_box,
				min_score_thresh=min_score_thresh,
				color=color,
				radius=line_thickness / 2,
				use_normalized_coordinates=use_normalized_coordinates,
				keypoint_edges=keypoint_edges,
				keypoint_edge_color=color,
				keypoint_edge_width=line_thickness // 2)

	return image, classes_list, scores_list, boxes_list

