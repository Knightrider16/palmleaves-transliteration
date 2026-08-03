from preprocessing_scripts.preprocess import run_preprocess
from preprocessing_scripts.batch_upscale import run_batch_upscale
from preprocessing_scripts.batch_postprocess import run_batch_postprocess
from preprocessing_scripts.batch_mask_clean import run_batch_mask_clean


def run_pipeline():

    print("\nSTEP 1: Preprocessing")
    run_preprocess()

    print("\nSTEP 2: Upscaling")
    run_batch_upscale()

    print("\nSTEP 3: Postprocessing")
    run_batch_postprocess()

    print("\nSTEP 4: Mask Cleaning")
    run_batch_mask_clean()

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()