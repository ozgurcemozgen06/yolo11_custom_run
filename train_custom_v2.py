from ultralytics import YOLO

def main():
    model = YOLO("yolo11s")  # no .pt file needed
    model.to("cpu")

    model.train(
        data="/Users/ozgurcem/yolo11_custom/custom_dataset.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        name="cli_train_fast", 
        project="results",
        save=True,
        save_period=10,
        workers=0,
        device="cpu"
    )

    # Perform test evaluation after training is fully done
    metrics = model.val(
        data="/Users/ozgurcem/yolo11_custom/custom_dataset.yaml",
        split="test",
        imgsz=640,
        device="cpu"
    )
    
    # Save test metrics (safe dictionary access)
    if hasattr(metrics, 'results_dict'):
        test_dict = metrics.results_dict
        with open("results/yolov11_custom_run/test_metrics.txt", "w") as f:
            for k, v in test_dict.items():
                f.write(f"{k}: {v:.4f}\n")

if __name__ == "__main__":
    main()
