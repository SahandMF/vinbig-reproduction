import json

input_path = "/media/arndt_ro/public_datasets/coco_lvis/annotations_coco/instances_train2017.json"
#input_path = "/media/arndt_ro/public_datasets/coco_lvis/annotations_coco/instances_val2017.json"
output_path = "/home/sahand/datasets/coco/annotations_coco/instances_train2017.json"

# Load the original annotation
with open(input_path, 'r') as f:
    data = json.load(f)

# Create mapping: original_cat_id → new_id (0 to len-1)
original_cat_ids = [cat['id'] for cat in data['categories']]
original_cat_ids_sorted = sorted(original_cat_ids)
cat_id_map = {old_id: new_id for new_id, old_id in enumerate(original_cat_ids_sorted)}

# Apply new IDs to categories
for cat in data['categories']:
    cat['id'] = cat_id_map[cat['id']]

# Apply new IDs to annotations
for ann in data['annotations']:
    ann['category_id'] = cat_id_map[ann['category_id']]

# Optional: verify
print("Original IDs:", original_cat_ids_sorted)
print("Mapped to:", list(cat_id_map.values()))

# Save updated annotation
with open(output_path, 'w') as f:
    json.dump(data, f)

print(f"✅ Saved remapped annotation to {output_path}")
