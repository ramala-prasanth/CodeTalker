import streamlit as st
import argparse
import demo  # Ensure demo.py is in the correct location
import os
import tempfile

# Define the generate_video function as provided
def generate_video(audio_path='./examples/photo.wav',
                   actor_file='./examples/thanos.ply',
                   save_path='./Demo',
                   model_path='./results/scantalk_masked_velocity_loss.pth.tar',
                   video_name='demo.mp4',
                   fps=30,
                   device='cuda'):
    # Create argument namespace
    args = argparse.Namespace(
        device=device,
        save_path=save_path,
        audio=audio_path,
        actor_file=actor_file,
        model_path=model_path,
        video_name=video_name,
        fps=fps,
        latent_channels=32,
        in_channels=3,
        out_channels=3,
        lstm_layers=3
    )

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'Meshes'))
        os.mkdir(os.path.join(save_path, 'Images'))

    # Call the main function from demo.py
    demo.generate_meshes(args)

    print('Starting Video Generation')
    demo.generate_mesh_video(save_path,
                        args.video_name,
                        os.path.join(save_path, 'Meshes'),
                        args.fps,
                        args.audio)

    return os.path.join(args.save_path, args.video_name)

# Streamlit app
st.title("Speech-Driven 3D Facial Animation App")

# Upload audio file
uploaded_audio = st.file_uploader("Upload an audio file", type=["wav"])

# Set actor file path (you can allow upload if needed)
actor_file = '/content/ScanTalk/src/examples/FLAME_sample.ply'  # Update the path as needed

# Set model path
model_path = '/content/ScanTalk/src/results/scantalk_masked_velocity_loss.pth.tar'  # Update the path as needed

# Set save path and video name
save_path = './output'
video_name = 'demo.mp4'

# Create a generate button
if st.button("Generate Video"):
    if uploaded_audio is not None:
        # Save the uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_audio.read())
            audio_path = temp_audio.name

        # Generate video with a spinner
        with st.spinner("Generating video..."):
            output_video_path = generate_video(
                audio_path=audio_path,
                actor_file=actor_file,
                model_path=model_path,
                save_path=save_path,
                video_name=video_name,
                fps=30,
                device='cuda'
            )

        # Display the video
        st.video(output_video_path)

        # Clean up the temporary audio file
        os.remove(audio_path)
    else:
        st.warning("Please upload an audio file.")
