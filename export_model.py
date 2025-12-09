import torch
import os
import argparse
import re
import glob
import onnx
from onnx.external_data_helper import load_external_data_for_model
from model import AlphaZeroNet
from config import Config

def get_latest_checkpoint():
    """Find the checkpoint with the highest iteration number across all checkpoint directories."""
    checkpoint_dirs = glob.glob('checkpoints*')
    all_checkpoints = []
    
    for d in checkpoint_dirs:
        if not os.path.isdir(d):
            continue
            
        files = [f for f in os.listdir(d) if f.endswith('.pth')]
        for f in files:
            match = re.search(r'model_iter_(\d+).pth', f)
            if match:
                iteration = int(match.group(1))
                full_path = os.path.join(d, f)
                all_checkpoints.append((iteration, full_path))
    
    if not all_checkpoints:
        return None
        
    # Sort by iteration descending
    all_checkpoints.sort(key=lambda x: x[0], reverse=True)
    return all_checkpoints[0][1]

def export_onnx(checkpoint_path, output_path):
    config = Config()
    device = torch.device('cpu')
    
    print(f"Initializing model with board_size={config.board_size}, num_channels={config.num_channels}, num_residual_blocks={config.num_residual_blocks}")
    model = AlphaZeroNet(
        board_size=config.board_size,
        num_channels=config.num_channels,
        num_residual_blocks=config.num_residual_blocks
    )
    
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # Infer model parameters from state_dict
        if 'conv.weight' in state_dict:
            num_channels = state_dict['conv.weight'].shape[0]
            print(f"Inferred num_channels: {num_channels}")
        else:
            num_channels = config.num_channels
            print(f"Could not infer num_channels, using config: {num_channels}")

        # Infer number of residual blocks
        num_residual_blocks = 0
        while True:
            key = f'residual_blocks.{num_residual_blocks}.conv1.weight'
            if key in state_dict:
                num_residual_blocks += 1
            else:
                break
        print(f"Inferred num_residual_blocks: {num_residual_blocks}")
        
        # Initialize model with inferred parameters
        model = AlphaZeroNet(
            board_size=config.board_size,
            num_channels=num_channels,
            num_residual_blocks=num_residual_blocks
        )
        
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.eval()
    
    # Create dummy input: (batch_size, channels, height, width)
    # AlphaZeroNet expects (batch_size, 1, board_size, board_size)
    dummy_input = torch.randn(1, 1, config.board_size, config.board_size, device=device)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export
    print(f"Exporting to {output_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['policy', 'value'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'policy': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            }
        )
        print("Export complete!")
        
        # Create embedded version
        print("Creating embedded model...")
        try:
            onnx_model = onnx.load(output_path)
            
            # Load external data if any
            load_external_data_for_model(onnx_model, base_dir=os.path.dirname(output_path))
            
            # Construct embedded path
            dir_name = os.path.dirname(output_path)
            base_name = os.path.basename(output_path)
            name, ext = os.path.splitext(base_name)
            embedded_output_path = os.path.join(dir_name, f"{name}_embedded{ext}")
            
            print(f"Saving embedded model to {embedded_output_path}")
            # onnx.save defaults to embedding data if model size allows (<2GB)
            onnx.save(onnx_model, embedded_output_path)
            print("Embedded export complete!")
            
        except Exception as e:
            print(f"Error creating embedded model: {e}")
            
    except Exception as e:
        print(f"Error exporting ONNX model: {e}")

def main():
    parser = argparse.ArgumentParser(description='Export AlphaZero model to ONNX')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint to export')
    parser.add_argument('--output', type=str, default='web/model.onnx', help='Output ONNX file path')
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = get_latest_checkpoint()
        
    if not checkpoint_path:
        print("No checkpoints found.")
        return

    export_onnx(checkpoint_path, args.output)

if __name__ == '__main__':
    main()

