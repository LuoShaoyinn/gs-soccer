import quadrants
import gc, time

quadrants.init(arch=quadrants.amdgpu, device_memory_GB=1)

def test_quadrant_leak(iterations=5):
    
    # Track the count before the loop
    gc.collect()
    q_array_list = list()
    
    print("malloc")
    for i in range(iterations):
        # 1. Create the object
        # Adjust arguments based on your specific Genesis/Quadrant setup
        q_array_list.append(quadrants.ndarray(shape=(1024, 1024, 1024, 1), dtype=float))
        
        # 2. Perform a dummy operation to ensure it's fully allocated
        q_array_list[-1].fill(1.0) 
        
    
    time.sleep(5)
    print("free")
    del q_array_list

    gc.collect()


for i in range(40):
    test_quadrant_leak()
    time.sleep(5)
time.sleep(5)
