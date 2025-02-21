import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO(r"C:\Users\Admin\Documents\yolov5\runs\segment\train2\weights\best.pt") 


class_names= ['car-part-crack', 'crack', 'detachment', 'flat-tire', 'glass-crack', 'lamp-crack', 'minor-deformation', 
              'moderate-deformation', 'paint-chips', 'scratch', 'scratches', 'severe-deformation', 'side-mirror-crack']
damage_insurance= {
    'car-part-crack': 800, 
    'crack': 700, 
    'detachment': 600, 
    'flat-tire': 1000, 
    'glass-crack': 1200, 
    'lamp-crack': 800, 
    'minor-deformation': 500, 
    'moderate-deformation': 700, 
    'paint-chips': 400, 
    'scratch': 300, 
    'scratches': 400, 
    'severe-deformation': 1500, 
    'side-mirror-crack': 800
}
output_dir = "stored_predictions"
os.makedirs(output_dir, exist_ok=True)

def apply_mask(image, masks):
    mask_overlay = image.copy()
    img_h, img_w, _ = image.shape 

    if masks is None or masks.data is None:
        return mask_overlay  

    masks = masks.data.cpu().numpy()  

    for mask in masks:
        mask = cv2.resize(mask, (img_w, img_h))  
        mask = (mask > 0.5).astype(np.uint8)  
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)  
        for c in range(3):
            mask_overlay[:, :, c] = np.where(mask == 1, mask_overlay[:, :, c] * 0.5 + color[c] * 0.5, mask_overlay[:, :, c])

    return mask_overlay

def predict_damage(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb)

    damage_info = {}
    total_insurance = 0
    
    for result in results:
        masks = result.masks  
        boxes = result.boxes
        class_ids = [int(box.cls[0]) for box in boxes]
        confidences = [float(box.conf[0]) for box in boxes]

        annotated_img = apply_mask(img_rgb, masks)
        
        for class_id, confidence in zip(class_ids, confidences):
            class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
            
            if class_name in damage_insurance:
                insurance_amount = damage_insurance[class_name]
                if class_name not in damage_info:
                    damage_info[class_name] = (insurance_amount, confidence)
                    total_insurance += insurance_amount

    predicted_img_save_path = os.path.join(output_dir, "segmented_image.jpg")
    cv2.imwrite(predicted_img_save_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    
    print("Detected Damages:")
    for damage_type, (insurance_amount, confidence) in damage_info.items():
        print(f"Damage Type: {damage_type}, Insurance Amount: ${insurance_amount}, Confidence: {confidence:.2f}")
    print(f"Total Insurance Amount: ${total_insurance}")

    plt.imshow(annotated_img)
    plt.axis('off')
    plt.title("Predicted Image with Segmentation Masks")
    plt.show()

test_image_path = r"C:\Users\Admin\Videos\carPartdamangeDetection\car13.jpg"
predict_damage(test_image_path)
