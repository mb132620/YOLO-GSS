import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8.yaml')


    model.train(data=r'',
              
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  
                batch=16,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
          
                amp=True,
                project='runs/train',
                name='exp',
                )