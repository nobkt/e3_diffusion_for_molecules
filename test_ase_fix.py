#!/usr/bin/env python3
"""
Test script to verify ASE database loading fix.
This script demonstrates that the fix resolves both issues:
1. Correct molecule count (1000, not 2000)
2. No tensor dimension errors with include_charges=False
"""

import sys
import subprocess
import time

def test_ase_loading():
    """Test ASE database loading with include_charges=False"""
    
    print("Testing ASE database loading fix...")
    print("=" * 50)
    
    # Run the command that was failing before
    cmd = [
        "python", "main_qm9.py",
        "--dataset", "ase_db",
        "--ase_db_path", "select.db", 
        "--n_epochs", "1",
        "--exp_name", "test_ase_fix",
        "--n_stability_samples", "100",
        "--diffusion_noise_schedule", "polynomial_2",
        "--diffusion_noise_precision", "1e-5",
        "--diffusion_steps", "1000", 
        "--diffusion_loss_type", "l2",
        "--batch_size", "4",  # Small batch for quick test
        "--nf", "64",         # Smaller network for quick test
        "--n_layers", "3",    # Fewer layers for quick test
        "--lr", "1e-4",
        "--test_epochs", "10",
        "--ema_decay", "0.9999",
        "--no_wandb",
        "--include_charges", "False"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    # Run process and capture output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Read output line by line for first few seconds
    start_time = time.time()
    output_lines = []
    
    while time.time() - start_time < 30:  # Run for 30 seconds max
        line = process.stdout.readline()
        if line:
            output_lines.append(line.strip())
            print(line.strip())
            
            # Check for success indicators
            if "Epoch: 0, iter:" in line and "Loss" in line:
                print("\n✅ SUCCESS: Training started successfully!")
                print("✅ No tensor dimension errors!")
                break
                
            # Check for molecule count
            if "Total molecules analyzed:" in line:
                if "1000" in line:
                    print("✅ SUCCESS: Correct molecule count (1000)")
                else:
                    print("❌ FAILED: Incorrect molecule count")
                    
        elif process.poll() is not None:
            break
    
    # Terminate the process
    process.terminate()
    process.wait()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    
    # Check for key success indicators in output
    output_text = "\n".join(output_lines)
    
    success_indicators = [
        ("Loaded 1000 molecules", "✅ Correct molecule count loaded"),
        ("Total molecules analyzed: 1000", "✅ Correct analysis count"),
        ("Epoch: 0, iter:", "✅ Training started successfully"),
    ]
    
    all_passed = True
    for indicator, message in success_indicators:
        if indicator in output_text:
            print(message)
        else:
            print(f"❌ FAILED: {message.replace('✅', '❌')}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The ASE database loading fix is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return all_passed

if __name__ == "__main__":
    success = test_ase_loading()
    sys.exit(0 if success else 1)