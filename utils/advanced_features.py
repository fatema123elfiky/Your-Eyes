from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


class AdvancedFeatures:


    CRITICAL_PRIORITY = {
        'car', 'bus', 'truck', 'train', 'motorcycle',
        'traffic light', 'stop sign'
    }

    HIGH_PRIORITY = {
        'person', 'bicycle', 'dog', 'cat'
    }

    MEDIUM_PRIORITY = {
        'chair', 'bench', 'couch', 'dining table', 'bed'
    }

    @staticmethod
    def categorize_by_priority(detected_objects: List[Dict]) -> Dict[str, List[Dict]]:
        categorized = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }

        for obj in detected_objects:
            label = obj['label']

            if label in AdvancedFeatures.CRITICAL_PRIORITY:
                categorized['critical'].append(obj)
            elif label in AdvancedFeatures.HIGH_PRIORITY:
                categorized['high'].append(obj)
            elif label in AdvancedFeatures.MEDIUM_PRIORITY:
                categorized['medium'].append(obj)
            else:
                categorized['low'].append(obj)

        return categorized

    @staticmethod
    def analyze_scene_regions(
        detected_objects: List[Dict],
        image_width: int,
        num_regions: int = 3
    ) -> Dict[str, List[Dict]]:
        if num_regions == 3:
            region_names = ['left', 'center', 'right']
            boundaries = [0, image_width / 3, 2 * image_width / 3, image_width]
        else:
            region_names = [f'region_{i}' for i in range(num_regions)]
            boundaries = [i * image_width / num_regions for i in range(num_regions + 1)]

        regions = {name: [] for name in region_names}

        for obj in detected_objects:
            center_x = obj['center_x']

            for i in range(len(boundaries) - 1):
                if boundaries[i] <= center_x < boundaries[i + 1]:
                    regions[region_names[i]].append(obj)
                    break

        return regions

    @staticmethod
    def generate_region_description(regions: Dict[str, List[Dict]]) -> str:
        descriptions = []

        for region_name, objects in regions.items():
            if not objects:
                continue

            labels = [obj['label'] for obj in objects]
            counts = Counter(labels)

            parts = []
            for label, count in counts.items():
                if count == 1:
                    parts.append(f"a {label}")
                else:
                    parts.append(f"{count} {label}s")

            if parts:
                region_desc = f"On your {region_name}, there is {', '.join(parts)}"
                descriptions.append(region_desc)

        return ". ".join(descriptions) + "." if descriptions else "No objects detected."

    @staticmethod
    def detect_potential_hazards(detected_objects: List[Dict]) -> List[Dict]:
        hazards = []

        for obj in detected_objects:
            label = obj['label']

            if label in AdvancedFeatures.CRITICAL_PRIORITY:
                hazard_info = obj.copy()
                hazard_info['hazard_level'] = 'critical'
                hazard_info['warning'] = f"Warning: {label} detected"
                hazards.append(hazard_info)

            elif label in AdvancedFeatures.HIGH_PRIORITY:
                if obj.get('area', 0) > 50000:
                    hazard_info = obj.copy()
                    hazard_info['hazard_level'] = 'high'
                    hazard_info['warning'] = f"Caution: {label} nearby"
                    hazards.append(hazard_info)

        return hazards

    @staticmethod
    def calculate_scene_complexity(detected_objects: List[Dict]) -> Dict[str, any]:
        if not detected_objects:
            return {
                'total_objects': 0,
                'unique_types': 0,
                'complexity_score': 0,
                'description': 'Empty scene'
            }

        total_objects = len(detected_objects)
        unique_types = len(set(obj['label'] for obj in detected_objects))

        complexity_score = min(100, (total_objects * 5) + (unique_types * 10))

        if complexity_score < 20:
            description = 'Simple scene'
        elif complexity_score < 50:
            description = 'Moderate scene'
        elif complexity_score < 75:
            description = 'Complex scene'
        else:
            description = 'Very complex scene'

        return {
            'total_objects': total_objects,
            'unique_types': unique_types,
            'complexity_score': complexity_score,
            'description': description
        }

    @staticmethod
    def suggest_navigation_guidance(
        detected_objects: List[Dict],
        image_width: int,
        image_height: int
    ) -> str:
        if not detected_objects:
            return "Path appears clear."

        regions = AdvancedFeatures.analyze_scene_regions(detected_objects, image_width)

        guidance = []

        center_objects = regions.get('center', [])
        if center_objects:
            critical_center = [obj for obj in center_objects
                             if obj['label'] in AdvancedFeatures.CRITICAL_PRIORITY]
            if critical_center:
                guidance.append("Stop! Obstacle directly ahead")
            else:
                guidance.append("Caution: objects ahead")

        left_objects = regions.get('left', [])
        if left_objects:
            guidance.append(f"{len(left_objects)} object(s) on your left")

        right_objects = regions.get('right', [])
        if right_objects:
            guidance.append(f"{len(right_objects)} object(s) on your right")

        if not guidance:
            return "Path appears clear."

        return ". ".join(guidance) + "."

