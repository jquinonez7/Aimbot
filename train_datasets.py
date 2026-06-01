#!/usr/bin/env python3
"""
Train Selected Datasets
Each dataset will be trained sequentially and results saved in respective folders.
"""

from ultralytics import YOLO
import time
import datetime
import os
import json
import yaml
from pathlib import Path

class SettingsManager:
    """Manage settings for training script"""
    def __init__(self, settings_file="settings/train_dataset.json"):
        self.settings_file = Path(settings_file)
        self.default_settings = {
            "default_dataset_folder": "dataset_yolo_format"
        }
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from file, create with defaults if doesn't exist"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                # Ensure all required keys exist
                for key, value in self.default_settings.items():
                    if key not in settings:
                        settings[key] = value
                return settings
            except (json.JSONDecodeError, IOError) as e:
                print(f"WARNING  Error loading settings: {e}")
                print("Using default settings.")
                return self.default_settings.copy()
        else:
            # Create settings file with defaults
            self.save_settings(self.default_settings)
            return self.default_settings.copy()
    
    def save_settings(self, settings=None):
        """Save settings to file"""
        if settings is None:
            settings = self.settings
        
        try:
            # Ensure settings directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"ERROR Error saving settings: {e}")
            return False
    
    def get_setting(self, key):
        """Get a specific setting value"""
        return self.settings.get(key, self.default_settings.get(key))
    
    def set_setting(self, key, value):
        """Set a specific setting value and save"""
        self.settings[key] = value
        return self.save_settings()
    
    def get_default_dataset_folder(self):
        """Get default dataset folder"""
        return self.get_setting("default_dataset_folder")
    
    def set_default_dataset_folder(self, folder):
        """Set default dataset folder"""
        return self.set_setting("default_dataset_folder", folder)

def check_config_file(config_path='settings/train_config.yaml'):
    """Check if training configuration file exists"""
    if Path(config_path).exists():
        print(f"SUCCESS Training configuration found: {config_path}")
        return config_path
    else:
        print(f"ERROR Configuration file {config_path} not found. YOLO will use default parameters.")
        return None

def manage_settings(settings_manager):
    """Manage training settings"""
    print("\n" + "="*60)
    print("SETTINGS MANAGEMENT")
    print("="*60)
    
    while True:
        print(f"\nCurrent settings:")
        print(f"  Default dataset folder: {settings_manager.get_default_dataset_folder()}")
        print(f"\nOptions:")
        print("1. Change default dataset folder")
        print("2. View current settings")
        print("3. Reset to defaults")
        print("4. Back to main menu")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                current_folder = settings_manager.get_default_dataset_folder()
                print(f"\nCurrent default dataset folder: {current_folder}")
                new_folder = input("Enter new default dataset folder path (or press Enter to cancel): ").strip()
                
                if new_folder:
                    # Validate folder exists
                    test_path = Path(new_folder)
                    if test_path.exists() and test_path.is_dir():
                        if settings_manager.set_default_dataset_folder(new_folder):
                            print(f"SUCCESS Default dataset folder updated to: {new_folder}")
                        else:
                            print("ERROR Failed to save settings")
                    else:
                        print(f"ERROR Folder does not exist: {new_folder}")
                        create = input("Create this folder? (y/n): ").strip().lower()
                        if create == 'y':
                            test_path.mkdir(parents=True, exist_ok=True)
                            if settings_manager.set_default_dataset_folder(new_folder):
                                print(f"SUCCESS Created folder and updated setting to: {new_folder}")
            
            elif choice == '2':
                print("\nCurrent Settings:")
                print(f"  Default dataset folder: {settings_manager.get_default_dataset_folder()}")
                input("\nPress Enter to continue...")
            
            elif choice == '3':
                confirm = input("Reset all settings to defaults? (y/n): ").strip().lower()
                if confirm == 'y':
                    settings_manager.settings = settings_manager.default_settings.copy()
                    if settings_manager.save_settings():
                        print("SUCCESS Settings reset to defaults")
            
            elif choice == '4':
                break
            
            else:
                print("ERROR Invalid choice. Please enter 1-4")
        
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled.")
            break
        except Exception as e:
            print(f"ERROR Error: {e}")

def get_available_models(models_dir='models'):
    """Get list of available model files from the models folder"""
    # Convert to absolute path if relative
    models_path = Path(models_dir)
    if not models_path.is_absolute():
        models_path = Path.cwd() / models_dir
    
    if not models_path.exists():
        print(f"ERROR Models directory not found: {models_path}")
        return []
    
    if not models_path.is_dir():
        print(f"ERROR Path is not a directory: {models_path}")
        return []
    
    model_files = list(models_path.glob('*.pt'))
    if not model_files:
        print(f"ERROR No .pt model files found in: {models_path}")
        return []
    
    # Return absolute paths
    return [str(model) for model in sorted(model_files)]

def get_model_selection():
    """Ask user to select a model from available models"""
    available_models = get_available_models()
    
    if not available_models:
        print("WARNING  No models found. Using default yolo11m.pt")
        return 'yolo11m.pt'
    
    while True:
        try:
            models_path = Path(available_models[0]).parent
            print(f"\nPACKAGE Available models in: {models_path}")
            print("-" * 60)
            for i, model in enumerate(available_models, 1):
                model_name = Path(model).name
                model_size = Path(model).stat().st_size / (1024 * 1024)  # Size in MB
                print(f"  {i}. {model_name} ({model_size:.1f} MB)")
            
            choice = input(f"\nSelect a model (1-{len(available_models)}): ").strip()
            
            if not choice:
                print("ERROR Please enter a valid number.")
                continue
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                selected_model = available_models[choice_idx]
                print(f"SUCCESS Selected model: {Path(selected_model).name}")
                return selected_model
            else:
                print(f"ERROR Please enter a number between 1 and {len(available_models)}")
                
        except ValueError:
            print("ERROR Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled by user.")
            return None

def get_dataset_folder():
    """Prompt user for the folder containing datasets"""
    while True:
        try:
            print("\n" + "="*60)
            print("DATASET FOLDER SELECTION")
            print("="*60)
            folder_input = input("Enter path to folder containing datasets (or press Enter for 'dataset_yolo_format'): ").strip()
            
            if not folder_input:
                folder_input = "dataset_yolo_format"
            
            dataset_folder = Path(folder_input)
            
            # Check if folder exists
            if not dataset_folder.exists():
                print(f"ERROR Folder does not exist: {dataset_folder}")
                create = input("Create this folder? (y/n): ").strip().lower()
                if create == 'y':
                    dataset_folder.mkdir(parents=True, exist_ok=True)
                    print(f"SUCCESS Created folder: {dataset_folder}")
                else:
                    continue
            
            if not dataset_folder.is_dir():
                print(f"ERROR Path is not a directory: {dataset_folder}")
                continue
            
            print(f"SUCCESS Using dataset folder: {dataset_folder.absolute()}")
            return str(dataset_folder)
            
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled by user.")
            return None
        except Exception as e:
            print(f"ERROR Invalid path: {e}")
            continue

def list_available_datasets(dataset_folder='dataset_yolo_format'):
    """List all available datasets in the folder"""
    dataset_folder_path = Path(dataset_folder)
    if not dataset_folder_path.exists():
        return []
    
    available_datasets = []
    
    # Check all subdirectories
    for item in dataset_folder_path.iterdir():
        if item.is_dir():
            # Check if it has a dataset.yaml file
            yaml_file = item / "dataset.yaml"
            if yaml_file.exists():
                # Try to determine the identifier
                folder_name = item.name
                if folder_name.startswith('dataset_'):
                    # Remove 'dataset_' prefix to get identifier
                    identifier = folder_name.replace('dataset_', '', 1)
                    # Try to convert to int if it's a number
                    try:
                        identifier = int(identifier)
                    except ValueError:
                        pass  # Keep as string
                else:
                    identifier = folder_name
                
                available_datasets.append(identifier)
    
    return sorted(available_datasets, key=lambda x: (isinstance(x, int), x))

def get_dataset_identifiers(dataset_folder='dataset_yolo_format'):
    """Get dataset identifiers (numbers or names) from user input"""
    # First, list all available datasets
    available_datasets = list_available_datasets(dataset_folder)
    
    if not available_datasets:
        print(f"ERROR No datasets found in {dataset_folder}")
        return None
    
    print("\n" + "="*60)
    print("AVAILABLE DATASETS")
    print("="*60)
    for i, dataset in enumerate(available_datasets, 1):
        # Display both the identifier and the folder name
        if isinstance(dataset, int):
            folder_name = f"dataset_{dataset}"
        else:
            folder_name = str(dataset)
        
        print(f"  {i}. {dataset} (folder: {folder_name})")
    print("="*60)
    print(f"\nTIP You can type 'all' to train all {len(available_datasets)} datasets")
    print("TIP Or enter specific dataset names/numbers separated by commas")
    
    while True:
        try:
            user_input = input("\nEnter dataset names/numbers (or 'all' for all datasets): ").strip()
            if not user_input:
                print("ERROR Please enter at least one dataset identifier or 'all'.")
                continue
            
            # Check if user wants all datasets
            if user_input.lower() == 'all':
                print(f"SUCCESS Selected all {len(available_datasets)} datasets")
                return available_datasets
            
            # Parse comma-separated identifiers (numbers or strings)
            dataset_identifiers = []
            for identifier in user_input.split(','):
                identifier = identifier.strip()
                if identifier:
                    # Try to convert to int if it's a number, otherwise keep as string
                    try:
                        dataset_identifiers.append(int(identifier))
                    except ValueError:
                        dataset_identifiers.append(identifier)
            
            if not dataset_identifiers:
                print("ERROR Please enter valid dataset identifiers.")
                continue
                
            # Remove duplicates while preserving order and mixed types
            seen = set()
            unique_identifiers = []
            for item in dataset_identifiers:
                if item not in seen:
                    seen.add(item)
                    unique_identifiers.append(item)
            
            # Validate that datasets exist
            valid_identifiers = []
            for identifier in unique_identifiers:
                identifier_str = str(identifier)
                found = False
                
                # If identifier already starts with 'dataset_', don't add it again
                if identifier_str.startswith('dataset_'):
                    # Try as-is first
                    dataset_path = Path(f"{dataset_folder}/{identifier_str}")
                    if dataset_path.exists():
                        valid_identifiers.append(identifier)
                        found = True
                    else:
                        # Also try without the prefix (in case folder name doesn't have dataset_)
                        dataset_path_no_prefix = Path(f"{dataset_folder}/{identifier_str.replace('dataset_', '', 1)}")
                        if dataset_path_no_prefix.exists():
                            valid_identifiers.append(identifier)
                            found = True
                else:
                    # Try with dataset_ prefix first (for numbered datasets like 1, 2, 3)
                    dataset_path = Path(f"{dataset_folder}/dataset_{identifier_str}")
                    if dataset_path.exists():
                        valid_identifiers.append(identifier)
                        found = True
                    else:
                        # Try without dataset_ prefix (for merged datasets, etc.)
                        dataset_path_no_prefix = Path(f"{dataset_folder}/{identifier_str}")
                        if dataset_path_no_prefix.exists():
                            valid_identifiers.append(identifier)
                            found = True
                
                if not found:
                    print(f"WARNING  Dataset '{identifier_str}' not found in {dataset_folder}/")
            
            if not valid_identifiers:
                print("ERROR No valid datasets found. Please check your dataset names/numbers.")
                continue
            
            print(f"SUCCESS Selected datasets: {', '.join(map(str, valid_identifiers))}")
            if len(valid_identifiers) != len(unique_identifiers):
                print(f" Note: Only {len(valid_identifiers)} out of {len(unique_identifiers)} datasets were found and will be used.")
            
            return valid_identifiers
            
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled by user.")
            return None

def get_report_name(num_datasets=1):
    """Get custom report name from user"""
    while True:
        try:
            if num_datasets == 1:
                user_input = input("\nEnter the name for the training run (or press Enter to use dataset name): ").strip()
            else:
                print("\n" + "="*60)
                print("TRAINING RUN NAMING")
                print("="*60)
                print("You are training multiple datasets separately.")
                print("Options:")
                print("  1. Press Enter to use dataset names (recommended)")
                print("  2. Enter a custom name to use the same name for all runs")
                user_input = input("\nEnter custom name or press Enter for dataset names: ").strip()
            
            if not user_input:
                print("SUCCESS Will use individual dataset names for each run")
                return None
            
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                user_input = user_input.replace(char, '_')
            
            if num_datasets == 1:
                print(f"SUCCESS Training results will be saved as: runs/detect/{user_input}/")
            else:
                print(f"SUCCESS All training results will be saved as: runs/detect/{user_input}/")
            return user_input
            
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled by user.")
            return None

def train_single_dataset(dataset_num, run_name="test", base_model='yolo11m.pt', config_path=None,
                        dataset_folder='dataset_yolo_format'):
    """Train a single dataset and return results"""
    print(f"\n{'='*60}")
    print(f"LAUNCH Starting training on Dataset {dataset_num}")
    print(f" Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MODEL Using: {base_model}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Load model
        model = YOLO(base_model)
        
        # Prepare basic training parameters
        # Check if dataset exists with or without dataset_ prefix
        dataset_path_with_prefix = Path(f'{dataset_folder}/dataset_{dataset_num}/dataset.yaml')
        dataset_path_without_prefix = Path(f'{dataset_folder}/{dataset_num}/dataset.yaml')
        
        if dataset_path_with_prefix.exists():
            dataset_yaml_path = dataset_path_with_prefix
            actual_dataset_folder = dataset_path_with_prefix.parent
        elif dataset_path_without_prefix.exists():
            dataset_yaml_path = dataset_path_without_prefix
            actual_dataset_folder = dataset_path_without_prefix.parent
        else:
            raise FileNotFoundError(
                f"Dataset not found in '{dataset_folder}': tried 'dataset_{dataset_num}' and '{dataset_num}'"
            )
        
        # Fix the path field in dataset.yaml if it's incorrect
        # YOLO uses the 'path' field to resolve image paths, so it must be correct
        try:
            with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
                yaml_data = yaml.safe_load(yaml_content)
            
            # Update path to point to the actual dataset folder
            correct_path = str(actual_dataset_folder.absolute())
            current_path = yaml_data.get('path', '')
            
            if current_path != correct_path:
                # Read original file to preserve formatting
                lines = yaml_content.split('\n')
                output_lines = []
                path_updated = False
                
                for line in lines:
                    if line.strip().startswith('path:') and not path_updated:
                        output_lines.append(f'path: {correct_path}')
                        path_updated = True
                    else:
                        output_lines.append(line)
                
                # Write back the corrected yaml
                with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(output_lines))
                print(f"SETTINGS  Updated dataset.yaml path: {current_path} → {correct_path}")
        except Exception as e:
            print(f"WARNING  Could not update dataset.yaml path: {e}")
            print(f"         Make sure the 'path' field in dataset.yaml points to: {actual_dataset_folder.absolute()}")
        
        train_params = {
            'data': str(dataset_yaml_path),
            'name': run_name
        }
        
        # Use config file if available, otherwise use basic defaults
        if config_path:
            print(f"LIST Using configuration from {config_path}")
            # Let YOLO load the config file directly
            train_params['cfg'] = config_path
        else:
            print("LIST Using YOLO default parameters")
            # Only set essential parameters if no config file
            train_params.update({
                'epochs': 50,
                'batch': 16,
                'imgsz': 640
            })
        
        # Train the model - YOLO will handle all the config parsing
        results = model.train(**train_params)
        
        # Calculate training time
        training_time = time.time() - start_time
        hours = training_time // 3600
        minutes = (training_time % 3600) // 60
        
        print(f"\nSUCCESS Dataset {dataset_num} training completed!")
        print(f"TIME  Training time: {int(hours)}h {int(minutes)}m")
        print(f"FOLDER Results saved in: runs/detect/{run_name}/")
        
        # Print final metrics
        try:
            final_results = results.results_dict
            map50 = final_results.get('metrics/mAP50(B)', 0)
            map50_95 = final_results.get('metrics/mAP50-95(B)', 0)
            print(f"CHART Final mAP50: {map50:.4f}")
            print(f"CHART Final mAP50-95: {map50_95:.4f}")
            
            return {
                'dataset': dataset_num,
                'success': True,
                'training_time': training_time,
                'map50': map50,
                'map50_95': map50_95,
                'model_used': base_model,
                'run_name': run_name
            }
        except:
            print("CHART Check results.csv for detailed metrics")
            return {
                'dataset': dataset_num,
                'success': True,
                'training_time': training_time,
                'map50': None,
                'map50_95': None,
                'model_used': base_model,
                'run_name': run_name
            }
            
    except Exception as e:
        print(f"ERROR Error training Dataset {dataset_num}: {e}")
        return {
            'dataset': dataset_num,
            'success': False,
            'error': str(e),
            'training_time': time.time() - start_time,
            'model_used': base_model,
            'run_name': run_name
        }

def main():
    print("TARGET AUTOMATED YOLO TRAINING WITH OPTIONS")
    print("=" * 60)
    print("TIP You can stop anytime with Ctrl+C")
    print("=" * 60)
    
    # Initialize settings manager
    settings_manager = SettingsManager()
    
    # Main menu loop
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Train datasets")
        print("2. Settings")
        print("3. Exit")
        
        try:
            menu_choice = input("\nSelect option (1-3): ").strip()
            
            if menu_choice == '1':
                # Start training workflow
                break
            elif menu_choice == '2':
                manage_settings(settings_manager)
                continue
            elif menu_choice == '3':
                print("EXIT Goodbye!")
                return
            else:
                print("ERROR Invalid choice. Please enter 1-3")
                continue
        except KeyboardInterrupt:
            print("\n\nEXIT Goodbye!")
            return
    
    # Check for training configuration file
    config_path = check_config_file()
    
    # Get dataset folder from user input
    dataset_folder = get_dataset_folder()
    if dataset_folder is None:
        print("ERROR Dataset folder selection cancelled.")
        return
    
    # Get user inputs
    dataset_identifiers = get_dataset_identifiers(dataset_folder)
    if dataset_identifiers is None:
        return
    
    base_model = get_model_selection()
    if base_model is None:
        return
    
    run_name = get_report_name(num_datasets=len(dataset_identifiers))
    # run_name can be None (use dataset names), which is valid
    
    # Display training plan
    print(f"\nLIST TRAINING PLAN:")
    print(f"CHART Datasets to train: {', '.join(map(str, dataset_identifiers))}")
    print(f"FOLDER Dataset folder: {dataset_folder}")
    print(f"MODEL Base model: {Path(base_model).name}")
    print(f"SETTINGS  Configuration: {config_path if config_path else 'YOLO defaults'}")
    
    if run_name:
        print(f"FOLDER Results will be saved in: runs/detect/{run_name}/")
    else:
        print(f"FOLDER Results will be saved in: runs/detect/dataset_[name]/ (separate for each)")
    
    print(f"TIME  Estimated time per dataset: 2-5 hours (depending on GPU and dataset size)")
    print("=" * 60)
    
    # Confirm before starting
    try:
        confirm = input("\nProceed with training? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("ERROR Training cancelled.")
            return
    except KeyboardInterrupt:
        print("\nERROR Training cancelled.")
        return
    
    results_summary = []
    overall_start_time = time.time()
    
    for i, dataset_identifier in enumerate(dataset_identifiers):
        try:
            # Determine the run name for this dataset
            if run_name:
                current_run_name = run_name
            else:
                current_run_name = f"dataset_{dataset_identifier}"
            
            result = train_single_dataset(
                dataset_num=dataset_identifier,
                run_name=current_run_name,
                base_model=base_model,
                config_path=config_path,
                dataset_folder=dataset_folder
            )
            results_summary.append(result)
            
            # Brief pause between trainings
            if i < len(dataset_identifiers) - 1:  # Not the last dataset
                print(f"\nPAUSE  Brief pause before next dataset...")
                time.sleep(5)
            
        except KeyboardInterrupt:
            print(f"\n\nSTOP Training interrupted by user!")
            print(f"CHART Completed datasets so far: {[r['dataset'] for r in results_summary if r.get('success', False)]}")
            break
        except Exception as e:
            print(f"\nERROR Unexpected error: {e}")
            continue
    
    # Print final summary
    total_time = time.time() - overall_start_time
    total_hours = total_time // 3600
    total_minutes = (total_time % 3600) // 60
    
    print(f"\n{'='*80}")
    print(f" TRAINING SUMMARY")
    print(f"{'='*80}")
    
    if run_name:
        print(f"CHART Run Name: {run_name}")
    else:
        print(f"CHART Run Names: Individual dataset names (dataset_[name])")
    
    print(f"MODEL Base Model: {Path(base_model).name}")
    print(f"SETTINGS  Configuration: {config_path if config_path else 'YOLO defaults'}")
    print(f"TIME  Total time: {int(total_hours)}h {int(total_minutes)}m")
    print(f"CHART Datasets processed: {len(results_summary)}")
    
    successful_trainings = [r for r in results_summary if r['success']]
    failed_trainings = [r for r in results_summary if not r['success']]
    
    print(f"SUCCESS Successful: {len(successful_trainings)}")
    print(f"ERROR Failed: {len(failed_trainings)}")
    
    if successful_trainings:
        print(f"\nMETRICS PERFORMANCE SUMMARY:")
        print(f"{'Dataset':<10} {'mAP50':<10} {'mAP50-95':<12} {'Time (min)':<12} {'Model Used':<20}")
        print(f"{'-'*70}")
        
        for result in successful_trainings:
            dataset = result['dataset']
            map50 = f"{result['map50']:.3f}" if result['map50'] else "N/A"
            map50_95 = f"{result['map50_95']:.3f}" if result['map50_95'] else "N/A"
            time_min = f"{result['training_time']/60:.0f}"
            model_used = result.get('model_used', 'unknown').split('/')[-1]  # Get filename only
            print(f"{dataset:<10} {map50:<10} {map50_95:<12} {time_min:<12} {model_used:<20}")
        
        # Identify best performing dataset
        valid_results = [r for r in successful_trainings if r['map50'] is not None]
        if valid_results:
            best_dataset = max(valid_results, key=lambda x: x['map50'])
            print(f"\nBEST Best performing dataset: Dataset {best_dataset['dataset']} (mAP50: {best_dataset['map50']:.3f})")
    
    if failed_trainings:
        print(f"\nERROR FAILED TRAININGS:")
        for result in failed_trainings:
            print(f"   Dataset {result['dataset']}: {result.get('error', 'Unknown error')}")
    
    if run_name:
        print(f"\nFOLDER All results saved in: runs/detect/{run_name}/")
    else:
        print(f"\nFOLDER Results saved in separate folders: runs/detect/dataset_[name]/")
        if successful_trainings:
            print(f"FOLDER Trained datasets:")
            for result in successful_trainings:
                print(f"       - runs/detect/{result['run_name']}/")
    
    print(f"SEARCH Check individual result folders for detailed metrics and visualizations")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
