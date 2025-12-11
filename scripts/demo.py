import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import gradio as gr
from PIL import Image
import scipy.io as sio
import numpy as np
import torch

import deblurganv2
import dncnn
import mair
import rednet
import restormer
from deblurganv2.models.fpn_inception import FPNInception
from deblurganv2.models.fpn_mobilenet import FPNMobileNet
from restormer import Restormer
from mair.basicsr.archs.mair_arch import MaIR
from mair.realDenoising.basicsr.models.archs.mairunet_arch import MaIRUNet
from utils import add_gaussian_noise, pad, run_model_inference
from configs import ROOT_DATASET_DIR, PATCH_CONFIG, ROOT_WEIGHTS_DIR


def get_task_data():
    task_data = {}

    for task in os.listdir(ROOT_DATASET_DIR):
        task_dir = os.path.join(ROOT_DATASET_DIR, task)
        if os.path.isdir(task_dir):
            subtasks = {}
            for subtask in os.listdir(task_dir):
                subtask_dir = os.path.join(task_dir, subtask)
                if os.path.isdir(subtask_dir):
                    datasets = []
                    for dataset in os.listdir(os.path.join(subtask_dir, 'test')):
                        datasets.append(dataset)

                    subtasks[subtask] = datasets

            task_data[task] = subtasks

    return task_data


def get_models(task, subtask, gray=False, blind_noise=False):
    model_dict = {
        'deblurring': {
            'defocus': ['Restormer'],
            'motion': ['DeblurGANv2 (Inception)', 'DeblurGANv2 (MobileNet)', 'Restormer', 'MaIR']
        },
        'denoising': {
            'gaussian': ['REDNet', 'DnCNN', 'Restormer', 'MaIR'],
            'real': ['Restormer', 'MaIR']
        },
    }
    models = model_dict.get(task, {}).get(subtask, [])

    if subtask == 'gaussian':
        if gray:
            if not blind_noise:
                if 'MaIR' in models:
                    models.remove('MaIR')
            else:
                if 'REDNet' in models:
                    models.remove('REDNet')
                if 'MaIR' in models:
                    models.remove('MaIR')
        else:
            if not blind_noise:
                if 'REDNet' in models:
                    models.remove('REDNet')
                if 'DnCNN' in models:
                    models.remove('DnCNN')
            else:
                if 'REDNet' in models:
                    models.remove('REDNet')
                if 'MaIR' in models:
                    models.remove('MaIR')

    return models


def get_patch_config(task, subtask, model_name) -> dict | None:
    task_key = task.lower()
    subtask_key = subtask.lower()
    model_key = model_name.split(' ')[0]
    config = PATCH_CONFIG.get(model_key, None)
    if isinstance(config, list):
        if model_key == 'DeblurGANv2':
            if 'Inception' in model_name:
                config = config[0]
            else:
                config = config[1]
        elif model_key == 'MaIR':
            if subtask_key == 'gaussian':
                config = config[0]
            else:
                config = config[1]
        elif model_key == 'Restormer':
            if task_key == 'denoising':
                config = config[0]
            else:
                config = config[1]
        else:
            config = config[0]

    return config


def get_model_instance(task, subtask, model_name, device, gray=False, sigma=None) -> torch.nn.Module:
    model_key = model_name.split(' ')[0]
    if model_key == 'REDNet':
        if task == 'denoising' and subtask == 'gaussian' and sigma is not None:
            return rednet.get_model(f'{ROOT_WEIGHTS_DIR}/REDNet/{sigma}.pt', device)
    elif model_key == 'DnCNN':
        if task == 'denoising' and subtask == 'gaussian':
            if gray:
                if sigma is not None:
                    return dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_{sigma}.pth', 1, 17, device)
                return dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_gray_blind.pth', 1, 20, device)
            if sigma is None:
                return dncnn.get_model(f'{ROOT_WEIGHTS_DIR}/DnCNN/dncnn_color_blind.pth', 3, 20, device)
    elif model_key == 'DeblurGANv2':
        if task == 'deblurring' and subtask == 'motion':
            if 'Inception' in model_name:
                return deblurganv2.get_model(f'{ROOT_WEIGHTS_DIR}/DeblurGANv2/fpn_inception.h5', device)
            if 'MobileNet' in model_name:
                return deblurganv2.get_model(f'{ROOT_WEIGHTS_DIR}/DeblurGANv2/fpn_mobilenet.h5', device)
    elif model_key == 'Restormer':
        if task == 'denoising':
            if subtask == 'gaussian':
                if sigma is not None:
                    return restormer.get_model(f"src/restormer/options/Gaussian{'Gray' if gray else 'Color'}Denoising_RestormerSigma{sigma}.yml", device)
                return restormer.get_model(f"src/restormer/options/Gaussian{'Gray' if gray else 'Color'}Denoising_Restormer.yml", device)
            if subtask == 'real':
                return restormer.get_model('src/restormer/options/RealDenoising_Restormer.yml', device)
        if task == 'deblurring':
            if subtask == 'defocus':
                return restormer.get_model('src/restormer/options/DefocusDeblur_Single_8bit_Restormer.yml', device)
            if subtask == 'motion':
                return restormer.get_model('src/restormer/options/Deblurring_Restormer.yml', device)
    elif model_key == 'MaIR':
        if task == 'denoising':
            if subtask == 'gaussian' and not gray and sigma is not None:
                return mair.get_model(f'src/mair/options/test_MaIR_CDN_s{sigma}.yml')
            if subtask == 'real':
                return mair.get_model('src/mair/realDenoising/options/test_MaIR_RealDN.yml')
        if task == 'deblurring' and subtask == 'motion':
            return mair.get_model('src/mair/realDenoising/options/test_MaIR_MotionDeblur.yml')

    raise ValueError('No model instance found for current configuration.')


def get_model_prediction(model, input_image, patch_size, patch_overlap, device, progress_bar=None):
    if isinstance(model, (FPNInception, FPNMobileNet)):
        restored_image, inference_time = run_model_inference(model,
                                                            input_image,
                                                            device,
                                                            normalize=deblurganv2.normalize,
                                                            pad=deblurganv2.pad,
                                                            postprocess=deblurganv2.postprocess,
                                                            patch_size=patch_size,
                                                            patch_overlap=patch_overlap,
                                                            progress_bar=progress_bar)
    elif isinstance(model, (Restormer, MaIR, MaIRUNet)):
        restored_image, inference_time = run_model_inference(model,
                                                            input_image,
                                                            device,
                                                            pad=pad,
                                                            patch_size=patch_size,
                                                            patch_overlap=patch_overlap,
                                                            progress_bar=progress_bar)
    else:
        restored_image, inference_time = run_model_inference(model,
                                                            input_image,
                                                            device,
                                                            patch_size=patch_size,
                                                            patch_overlap=patch_overlap,
                                                            progress_bar=progress_bar)
    return restored_image, inference_time


def update_subtask(task):
    task_key = task.lower()
    choices = list(task_data.get(task_key, {}).keys())
    title_choices = [c.title() for c in choices]
    return gr.update(choices=title_choices, value=title_choices[0])


def update_dataset(task, subtask):
    task_key = task.lower()
    subtask_key = subtask.lower()
    choices = task_data.get(task_key, {}).get(subtask_key, [])
    return (
        gr.update(choices=choices, value=choices[0]),
        gr.update(interactive=(subtask_key == 'gaussian'))
    )


def update_samples(task, subtask, dataset, n_samples=10):
    task_key = task.lower()
    subtask_key = subtask.lower()

    if task_key == 'deblurring':
        if dataset == 'DPDD':
            input_subdir = 'inputC'
        else:
            input_subdir = 'input'
        dir_path = os.path.join(ROOT_DATASET_DIR, task_key, subtask_key, 'test', dataset, input_subdir)
    else:
        dir_path = os.path.join(ROOT_DATASET_DIR, task_key, subtask_key, 'test', dataset)

    if dataset == 'SIDD':
        mat = sio.loadmat(os.path.join(dir_path, 'ValidationNoisyBlocksSrgb.mat'))
        img_array = mat['ValidationNoisyBlocksSrgb']
        N, M = img_array.shape[0], img_array.shape[1]
        images = [img_array[i, j, ...] for i in range(N) for j in range(M)]
        images = random.sample(images, n_samples)
        return map(Image.fromarray, images)

    image_files = sorted([os.path.join(dir_path, f)
                        for f in random.sample(os.listdir(dir_path), n_samples)
                        if os.path.isfile(os.path.join(dir_path, f))])
    return map(Image.open, image_files)


def update_models(task, subtask, dataset, input_image, blind_noise=False):
    task_key = task.lower()
    subtask_key = subtask.lower()
    if input_image is not None:
        gray = np.all(np.diff(input_image, axis=2) == 0)
    else:
        gray = dataset in ['Set12', 'BSD68']

    models = get_models(task_key, subtask_key, gray, blind_noise)
    return gr.update(choices=models, value=models[0])


def update_noisy_image(image, sigma):
    gray = np.all(np.diff(image, axis=2) == 0)
    # with gray image, keep only one channel
    if gray:
        image = image[:, :, :1]

    noisy_img = add_gaussian_noise(image, sigma)
    noisy_img = np.clip(noisy_img * 255, 0, 255).astype(np.uint8)

    # repeat channels to make it 3-channel image
    if gray:
        noisy_img = np.repeat(noisy_img, 3, axis=2)

    # return noisy image and set the added-noise flag (state) to True
    return Image.fromarray(noisy_img), True


def show_selected(input_source, images, evt: gr.SelectData):
    if input_source == 'Upload Image':
        return None

    selected = images[evt.index]
    return selected[0]


def update_input_image(image, subtask, added_noise_state):
    if image is None:
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            False
        )

    subtask_key = subtask.lower()
    if subtask_key == 'gaussian':
        if added_noise_state:
            return (
                gr.update(interactive=False),
                gr.update(interactive=True),
                False
            )

        return (
            gr.update(interactive=True),
            gr.update(interactive=False),
            False
        )

    return (
        gr.update(interactive=False),
        gr.update(interactive=True),
        False
    )


def update_patch_config(task, subtask, model_name):
    config = get_patch_config(task, subtask, model_name)
    if config is not None:
        return (
            gr.update(value=config['patch_size']),
            gr.update(value=config['patch_overlap'])
        )

    return (
        gr.update(value=None),
        gr.update(value=None)
    )


def run_restoration(input_image, task, subtask, model_name, patch_size, patch_overlap, blind_noise, sigma, device,progress=gr.Progress()):
    task_key = task.lower()
    subtask_key = subtask.lower()
    gray = np.all(np.diff(input_image, axis=2) == 0)
    sigma_value = None if blind_noise or subtask_key == 'real' else sigma
    device = torch.device(device)
    model = get_model_instance(task_key, subtask_key, model_name, device, gray, sigma_value)
    if gray:
        input_image = input_image[:, :, :1]
        restored_image, _ = get_model_prediction(model, input_image, patch_size, patch_overlap, device, progress)
        restored_image = np.repeat(restored_image, 3, axis=2)
    else:
        restored_image, _ = get_model_prediction(model, input_image, patch_size, patch_overlap, device, progress)
    return restored_image


def update_results(result_images, input_image, output_image, left_source, right_source):
    if result_images is None:
        result_images = []

    result_images.append(output_image)

    return (
        gr.update(value=result_images),
        update_compare_image(left_source, input_image),
        update_compare_image(right_source, output_image)
    )

def update_compare_image(source, used_image):
    if source == 'Input Image' or source == 'Restored Image':
        return gr.update(value=used_image, interactive=False)

    if source == 'Upload Image':
        return gr.update(value=None, interactive=True)

    return gr.update(value=None, interactive=False)


def update_image_slider(img1, img2):
    if img1 is None and img2 is None:
        return gr.update(value=None)

    return gr.update(value=(img1, img2))


def select_from_sample(left_src, right_src, active, images, evt: gr.SelectData):
    selected = images[evt.index]
    img = selected[0]
    left_update, right_update = gr.update(), gr.update()
    if active == 'left' and left_src == 'Sample Images':
        left_update = gr.update(value=img, interactive=False)
    elif active == 'right' and right_src == 'Sample Images':
        right_update = gr.update(value=img, interactive=False)
    return left_update, right_update


def select_from_results(left_src, right_src, active, images, evt: gr.SelectData):
    selected = images[evt.index]
    img = selected[0]
    left_update, right_update = gr.update(), gr.update()
    if active == 'left' and left_src == 'Result Images':
        left_update = gr.update(value=img, interactive=False)
    elif active == 'right' and right_src == 'Result Images':
        right_update = gr.update(value=img, interactive=False)
    return left_update, right_update


title = 'Image Restoration Demo'

task_data = get_task_data()
initial_tasks = list(task_data.keys())
initial_subtasks = list(task_data[initial_tasks[0]].keys())
initial_datasets = task_data[initial_tasks[0]][initial_subtasks[0]]
initial_images = update_samples(initial_tasks[0], initial_subtasks[0], initial_datasets[0])
initial_models = get_models(initial_tasks[0], initial_subtasks[0], initial_datasets[0] in ['Set12', 'BSD68'])
initial_patch_config = get_patch_config(initial_tasks[0], initial_subtasks[0], initial_models[0])

with gr.Blocks(title=title) as demo:
    gr.HTML(f"<center><h1>{title}</h1></center><br>")

    gr.Markdown("## Select Task")
    with gr.Column():
        with gr.Row():
            task_dropdown = gr.Dropdown(choices=map(str.title, initial_tasks),
                                        label='Task',
                                        interactive=True,
                                        value=initial_tasks[0].title())
            subtask_dropdown = gr.Dropdown(choices=map(str.title, initial_subtasks),
                                            label='Sub-Task',
                                            interactive=True,
                                            value=initial_subtasks[0].title())
            dataset_dropdown = gr.Dropdown(choices=initial_datasets,
                                            label='Dataset',
                                            interactive=True,
                                            value=initial_datasets[0])

        sample_images = gr.Gallery(value=initial_images, label='Sample Images', columns=5, format='png')

    gr.Markdown("## Run Restoration")
    with gr.Row():
        with gr.Column():
            input_source = gr.Radio(choices=['Upload Image', 'Select from Samples'],
                                    label='Input Source',
                                    value='Upload Image',
                                    interactive=True)

            with gr.Group():
                blind_noise_checkbox = gr.Checkbox(label='Blind Noise Level',
                                                   info='Check this box if the noise level is unknown by the model.',
                                                   value=False)

                with gr.Row():
                    sigma_dropdown = gr.Dropdown(choices=[15, 25, 50],
                                                label='Noise Level (Sigma)',
                                                info='Select noise level which models are trained for.',
                                                interactive=True,
                                                value=25)
                    sigma_slider = gr.Slider(minimum=15,
                                            maximum=50,
                                            step=5,
                                            label='Noise Level (Sigma)',
                                            info='Select noise level to add Gaussian noise to the input image.',
                                            interactive=False,
                                            value=25)

                add_noise_btn = gr.Button('Add Gaussian Noise', interactive=False)

        with gr.Column():
            with gr.Group():
                with gr.Row():
                    model_dropdown = gr.Dropdown(choices=initial_models,
                                                label='Model',
                                                interactive=True,
                                                value=initial_models[0])
                    device_dropdown = gr.Dropdown(choices=['cuda', 'cpu'],
                                                  label='Device',
                                                  interactive=True,
                                                  value='cuda' if torch.cuda.is_available() else 'cpu')

                patch_size_slider = gr.Slider(minimum=128,
                                            maximum=2048,
                                            step=32,
                                            label='Patch Size',
                                            info='Reduce patch size if you encounter out-of-memory errors.',
                                            interactive=True,
                                            value=initial_patch_config['patch_size'])
                patch_overlap_slider = gr.Slider(minimum=0,
                                                maximum=384,
                                                step=16,
                                                label='Patch Overlap',
                                                interactive=True,
                                                value=initial_patch_config['patch_overlap'])

            run_btn = gr.Button('Run', variant='primary', interactive=False)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label='Input Image', interactive=True, format='png')


        with gr.Column():
            output_image = gr.Image(label='Restored Image', interactive=False)

    result_images = gr.Gallery(label='Result Images', columns=5)

    gr.Markdown("## Compare Result")
    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    left_source = gr.Radio(choices=['Upload Image', 'Input Image',
                                                    'Sample Images', 'Result Images'],
                                            label='Left Source',
                                            value='Input Image',
                                            info=('Re-select this radio to activate this side,'
                                                  'then pick an image from the Sample/Result Gallery.'),
                                            interactive=True)

                    left_image = gr.Image(label='Left Image', interactive=False)

            with gr.Column():
                with gr.Group():
                    right_source = gr.Radio(choices=['Upload Image', 'Restored Image',
                                                     'Sample Images', 'Result Images'],
                                            label='Right Source',
                                            value='Restored Image',
                                            info=('Re-select this radio to activate this side,'
                                                  'then pick an image from the Sample/Result Gallery.'),
                                            interactive=True)

                    right_image = gr.Image(label='Right Image', interactive=False)

        image_slider = gr.ImageSlider(label='Image Slider', interactive=False)
        swap_btn = gr.Button('Swap Images')

    # state to remember which compare side was last activated (clicked)
    active_side = gr.State('left')
    # state to remember whether gaussian noise was added to the input image
    added_noise = gr.State(False)


    # Set up interactions
    task_dropdown.change(update_subtask,
                         inputs=[task_dropdown],
                         outputs=[subtask_dropdown])

    subtask_dropdown.change(update_dataset,
                            inputs=[task_dropdown, subtask_dropdown],
                            outputs=[dataset_dropdown, add_noise_btn])

    dataset_dropdown.change(update_samples,
                            inputs=[task_dropdown, subtask_dropdown, dataset_dropdown],
                            outputs=[sample_images])

    dataset_dropdown.change(update_models,
                            inputs=[task_dropdown, subtask_dropdown, dataset_dropdown, input_image, blind_noise_checkbox],
                            outputs=[model_dropdown])

    input_image.change(update_models,
                        inputs=[task_dropdown, subtask_dropdown, dataset_dropdown, input_image, blind_noise_checkbox],
                        outputs=[model_dropdown])

    add_noise_btn.click(update_noisy_image,
                        inputs=[input_image, sigma_slider],
                        outputs=[input_image, added_noise])

    sample_images.select(fn=show_selected,
                         inputs=[input_source, sample_images],
                         outputs=[input_image])

    sample_images.select(fn=select_from_sample,
                         inputs=[left_source, right_source, active_side, sample_images],
                         outputs=[left_image, right_image])

    input_source.change(fn=lambda src: gr.update(interactive=(src == 'Upload Image')),
                        inputs=[input_source],
                        outputs=[input_image])

    blind_noise_checkbox.change(fn=lambda blind, task, subtask, dataset, input_image: (
                                    gr.update(interactive=not blind),
                                    gr.update(interactive=blind),
                                    update_models(task, subtask, dataset, input_image, blind)
                                ),
                                inputs=[blind_noise_checkbox, task_dropdown, subtask_dropdown, dataset_dropdown, input_image],
                                outputs=[sigma_dropdown, sigma_slider, model_dropdown])

    input_image.change(fn=update_input_image,
                        inputs=[input_image, subtask_dropdown, added_noise],
                        outputs=[add_noise_btn, run_btn, added_noise])

    model_dropdown.change(update_patch_config,
                          inputs=[task_dropdown, subtask_dropdown, model_dropdown],
                          outputs=[patch_size_slider, patch_overlap_slider])

    run_btn.click(run_restoration,
                  inputs=[input_image, task_dropdown, subtask_dropdown,
                          model_dropdown, patch_size_slider, patch_overlap_slider,
                          blind_noise_checkbox, sigma_dropdown, device_dropdown],
                  outputs=[output_image])

    output_image.change(fn=update_results,
                        inputs=[result_images, input_image, output_image,
                                left_source, right_source],
                        outputs=[result_images, left_image, right_image])

    left_source.select(fn=update_compare_image,
                       inputs=[left_source, input_image],
                       outputs=[left_image])

    right_source.select(fn=update_compare_image,
                        inputs=[right_source, output_image],
                        outputs=[right_image])

    left_source.select(fn=lambda: 'left',
                        inputs=[],
                        outputs=[active_side])

    right_source.select(fn=lambda: 'right',
                        inputs=[],
                        outputs=[active_side])

    result_images.select(fn=select_from_results,
                         inputs=[left_source, right_source, active_side, result_images],
                         outputs=[left_image, right_image])

    left_image.change(fn=update_image_slider,
                      inputs=[left_image, right_image],
                      outputs=[image_slider])

    right_image.change(fn=update_image_slider,
                       inputs=[left_image, right_image],
                       outputs=[image_slider])

    swap_btn.click(fn=lambda imgs: imgs[::-1],
                   inputs=[image_slider],
                   outputs=[image_slider])


demo.launch(theme=gr.themes.Origin())
