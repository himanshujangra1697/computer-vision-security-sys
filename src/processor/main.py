import sys
from processor.inference_engine import HaloCoreEngine

def main():
    print("====================================")
    print("   HALOCORE EDGE INFERENCE v1.5     ")
    print("====================================")
    
    try:
        # Boot up the engine (GPU, Database, Camera)
        engine = HaloCoreEngine()
        
        # Start the continuous inference and hardware loop
        engine.run()
        
    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard interrupt detected. Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] Engine crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()