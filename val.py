import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.val(data=r'',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, 
              project='runs/val',
              name='exp',
              )