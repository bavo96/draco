# Download BIB recognition's weights
mkdir draco/models/BIB_recognition
wget -O draco/models/BIB_recognition/weights.zip https://storage.googleapis.com/marketplace_ai_project/adult_content_recognition/BIB_detection/BIB_model/50k_full/weights.zip
unzip draco/models/BIB_recognition/weights.zip -d draco/models/BIB_recognition/
rm -rf draco/models/BIB_recognition/weights.zip

# Download BIB board detection's weights
mkdir draco/models/BIB_board_detection
wget -O draco/models/BIB_board_detection/weights.zip https://storage.googleapis.com/marketplace_ai_project/adult_content_recognition/BIB_detection/BIB_board_model/weights.zip
unzip draco/models/BIB_board_detection/weights.zip -d draco/models/BIB_board_detection/
rm -rf draco/models/BIB_board_detection/weights.zip