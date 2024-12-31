import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

# Load the YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Public/pythonProject3/yolov5/yolov5s.pt').to(device)  # Ensure the model is loaded on the right device

# Define vehicle classes and their descriptions
CLASSES = ['tank', 'light utility vehicle', 'armored personnel carrier', 'infantry fighting vehicle',
           'anti-aircraft', 'self-propelled artillery', 'mine-protected vehicle', 'armored combat support vehicle',
           'prime movers and trucks', 'light armored vehicle']

VEHICLE_DESCRIPTIONS = {
    'tank': ("Tanks are heavily armored fighting vehicles equipped with powerful cannons and machine guns. "
             "Designed primarily for frontline combat, tanks are built to take on enemy forces with direct firepower "
             "while providing protection for their crew. They have thick armor that can withstand high-caliber rounds "
             "and explosive devices. Tanks are often supported by infantry and other vehicles, forming the backbone of "
             "armored ground forces. Modern tanks also feature advanced targeting systems, allowing them to engage targets "
             "at long ranges with high precision."),

    'light utility vehicle': ("Light utility vehicles, commonly known as light tactical vehicles, serve a wide range of roles in both "
                              "military and civilian applications. These versatile vehicles, often equipped with all-terrain capabilities, "
                              "can be used for troop transport, reconnaissance, or supply delivery in combat zones. Their speed and mobility "
                              "allow for quick deployment in various environments, including desert, forest, and urban settings. Though not "
                              "heavily armored, many can be retrofitted with armor plating or defensive weaponry to enhance protection."),

    'armored personnel carrier': ("Armored Personnel Carriers (APCs) are designed to transport soldiers and equipment safely across hostile "
                                  "territories. Unlike tanks, APCs are not intended for direct combat but provide mobility and protection from "
                                  "small arms fire and artillery shrapnel. With space for infantry and often equipped with a turret-mounted weapon, "
                                  "APCs are essential for moving troops into battle zones or evacuating the wounded. Their robust design and all-terrain "
                                  "capabilities allow them to traverse difficult landscapes."),

    'infantry fighting vehicle': ("Infantry Fighting Vehicles (IFVs) are a hybrid between tanks and armored personnel carriers, designed to support "
                                  "infantry operations while providing firepower. IFVs typically feature heavier armament than APCs, including cannons, "
                                  "anti-tank guided missiles, and machine guns. They can engage enemy forces independently or in coordination with tanks, "
                                  "giving infantry units the fire support they need to advance on enemy positions. Modern IFVs are highly mobile and capable "
                                  "of operating in a variety of environments."),

    'anti-aircraft': ("Anti-aircraft vehicles are specialized platforms equipped with high-caliber automatic cannons or surface-to-air missile systems. "
                      "They are designed to track and engage enemy aircraft, helicopters, and drones, providing protection for ground forces. Advanced "
                      "radar and targeting systems allow these vehicles to accurately engage aerial threats even at high altitudes or speeds. These vehicles "
                      "play a critical role in modern warfare by ensuring air superiority and protecting critical infrastructure from aerial attacks."),

    'self-propelled artillery': ("Self-propelled artillery units combine the long-range firepower of traditional artillery with the mobility of an armored vehicle. "
                                 "These vehicles can move quickly on the battlefield, deploy rapidly, and engage targets from a distance with precision-guided munitions. "
                                 "Typically mounted on tracked or wheeled platforms, self-propelled artillery can fire shells over large distances, supporting ground forces "
                                 "by bombarding enemy positions. Their ability to relocate swiftly makes them less vulnerable to counter-battery fire."),

    'mine-protected vehicle': ("Mine-protected vehicles are engineered to withstand explosions from land mines and improvised explosive devices (IEDs), which are "
                               "a common threat in modern conflict zones. These vehicles are heavily armored and feature a V-shaped hull to deflect blasts away from "
                               "the occupants. They are used primarily for troop transport, medical evacuation, or logistical support in high-risk areas. Mine-protected "
                               "vehicles have become an essential part of military operations, especially in regions where guerrilla warfare and ambush tactics are prevalent."),

    'armored combat support vehicle': ("Armored combat support vehicles provide logistical and operational support to frontline units. These vehicles are used for a variety "
                                       "of roles including ammunition supply, field repairs, medical evacuation, and command and control functions. While not intended for direct "
                                       "combat, they are often armored to withstand small arms fire and explosive devices. These vehicles are essential for maintaining the operational "
                                       "effectiveness of combat units, ensuring that troops have the supplies, communication, and medical support needed to sustain prolonged engagements."),

    'prime movers and trucks': ("Prime movers and trucks play a crucial role in transporting heavy equipment, artillery, and supplies on the battlefield. These vehicles are essential "
                                "for moving large loads, such as tanks, construction equipment, or ammunition, over long distances. Often equipped with off-road capabilities, prime movers "
                                "can traverse rough terrain while carrying significant weight. In military logistics, they ensure that vital resources and weaponry reach the frontlines in a timely manner."),

    'light armored vehicle': ("Light Armored Vehicles (LAVs) offer a balance between mobility, protection, and firepower. These wheeled vehicles are faster and more versatile than tracked "
                              "armored vehicles, making them ideal for reconnaissance missions, patrol duties, and rapid response operations. LAVs are often equipped with small to medium-caliber "
                              "weapons, such as machine guns or autocannons, and can be outfitted with additional armor to protect against small arms fire. Their speed and adaptability make them a "
                              "valuable asset in both combat and peacekeeping missions.")
}

def upload_video_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    video_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    return video_path

def detect_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Dictionary to store detected vehicles
    detected_vehicles = set()

    frame_count = 0  # To keep track of frame number

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video or error.")
            break

        frame_count += 1

        # Process every 5th frame to optimize processing speed (adjust N as needed)
        if frame_count % 5 == 0:
            # Perform object detection
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for torch model compatibility
            results = model(frame)

            # Get bounding boxes and class labels
            detections = results.xyxy[0].cpu().numpy()

            for detection in detections:
                _, _, _, _, _, cls = detection
                class_id = int(cls)

                # Check if class_id is valid
                if 0 <= class_id < len(CLASSES):
                    label = CLASSES[class_id]

                    # Add detected vehicle type to the set
                    detected_vehicles.add(label)

                    # Draw bounding boxes and class labels
                    cv2.rectangle(frame, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label}', (int(detection[0]), int(detection[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)

            # Convert frame back to BGR for OpenCV display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Show the frame with detection
            cv2.imshow('Military Vehicle Detection - Video', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Show the vehicle names and descriptions in a tkinter window
    show_vehicle_descriptions(detected_vehicles)

# Function to show vehicle names and descriptions in a tkinter window
def show_vehicle_descriptions(vehicles):
    window = tk.Tk()
    window.title("Detected Vehicle Descriptions")

    # Create a scrollbar in case of long content
    scrollbar = tk.Scrollbar(window)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    text_area = tk.Text(window, wrap=tk.WORD, yscrollcommand=scrollbar.set)
    text_area.pack(expand=True, fill='both')

    # Add the detected vehicle names and descriptions to the text area
    for vehicle in vehicles:
        text_area.insert(tk.END, f"{vehicle.upper()}:\n")
        text_area.insert(tk.END, f"{VEHICLE_DESCRIPTIONS[vehicle]}\n\n")

    scrollbar.config(command=text_area.yview)
    window.mainloop()

# Main function to handle video input
def main():
    video_path = upload_video_file()
    if video_path:  # Proceed only if a file was selected
        detect_in_video(video_path)
    else:
        print("No video file selected.")

if __name__ == "__main__":
    main()
