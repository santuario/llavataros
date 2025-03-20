import numpy as np
import pyvista as pv
import subprocess
import os
import tqdm
from scipy.spatial.transform import Rotation

'''
Uses PyVista and system ffmpeg to render some points (connected in a pseudo-skeleton),
joins the render together with a WAV file.
'''

def render_3d_animation_with_audio(args):
    """
    Render a 3D skeleton animation with accompanying audio using PyVista.
    
    Parameters:
    args : object with attributes:
        point_path : str
            Path to NPY file containing point data, shape (frames, points, 7)
            Each point has [px, py, pz, qx, qy, qz, qw]
        wav_path : str
            Path to the WAV audio file
        out_path : str
            Path for the output MOV file
        fps : float
            Frames per second for the animation
    """
    # Load point data
    point_data = np.load(args.point_path)
    num_frames, num_points, attr_dim = point_data.shape
    
    if attr_dim != 7:
        raise ValueError(f"Expected point data with 7 attributes (px, py, pz, qx, qy, qz, qw), got {attr_dim}")
    if num_points != 9:
        print(f"Warning: Expected 9 points, got {num_points}. Skeleton connections may need adjustment.")

    # Temporary video file
    temp_video = 'temp_video.mp4'
    
    # Skeleton connection topology (based on 9 points: root, hips, spine, etc.)
    skeleton_connections = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]  # Adjust as needed
    
    # Create plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background('black')
    
    # Create spheres for points
    spheres = []
    for i in range(num_points):
        sphere = pv.Sphere(radius=0.02)
        spheres.append(plotter.add_mesh(sphere, color='lightgreen'))
    
    # Create lines for skeleton connections
    lines = []
    for start, end in skeleton_connections:
        line = pv.Line()
        lines.append(plotter.add_mesh(line, color='gray', line_width=7))
    
    # Create text 3D actors for each point
    text_actors = []
    
    # Render frames
    plotter.open_movie(temp_video, framerate=args.fps)
    
    for frame in tqdm.tqdm(range(num_frames)):
        # Remove previous text actors
        for actor in text_actors:
            plotter.remove_actor(actor)
        text_actors = []
        
        # Extract positions and quaternions
        points = point_data[frame, :, :3]  # [px, py, pz]
        quaternions = point_data[frame, :, 3:]  # [qx, qy, qz, qw]
        
        # Calculate center of mass (using the root point, index 0)
        center_of_mass = points[0]
        
        # Calculate bounding box to determine scale
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        bbox_diagonal = np.linalg.norm(max_point - min_point)
        
        # Set camera position
        distance = max(bbox_diagonal * 1.5, 5)  # Ensure minimum distance of 5
        side_offset = distance * 0.5
        
        plotter.camera_position = [
            (center_of_mass[0] + side_offset, center_of_mass[1], center_of_mass[2] + side_offset * 1.5),
            (center_of_mass[0], center_of_mass[1], center_of_mass[2]),
            (0, 0, 1)  # Up vector
        ]
        plotter.camera.clipping_range = (0.1, 100000)
        
        # Update spheres with positions
        for i, sphere_actor in enumerate(spheres):
            center = points[i]
            sphere = pv.Sphere(radius=0.02, center=center)
            sphere_actor.GetMapper().SetInputData(sphere)
        
        # Update skeleton connections
        for (line_actor, (start, end)) in zip(lines, skeleton_connections):
            start_point = points[start]
            end_point = points[end]
            line = pv.Line(start_point, end_point)
            line_actor.GetMapper().SetInputData(line)
        
        # Add index labels to points
        for i, point in enumerate(points):
            text_pos = point + np.array([0.03, 0.03, 0.03])
            text = pv.Text3D(str(i), depth=0.01)
            text.points = text.points * 0.05
            text.points = text.points + text_pos
            text_actor = plotter.add_mesh(text, color='white')
            text_actors.append(text_actor)
        
        # Render frame
        plotter.write_frame()
    
    # Close movie
    plotter.close()
    
    # Combine video and audio using FFmpeg
    try:
        ffmpeg_command = [
            'ffmpeg',
            '-i', temp_video,
            '-i', args.wav_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            '-y',
            args.out_path
        ]
        subprocess.run(ffmpeg_command, check=True)
        os.remove(temp_video)
        print(f"3D Animation with audio saved to {args.out_path}")
    
    except subprocess.CalledProcessError as e:
        print(f"Error combining video and audio: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--point_path', type=str, required=True, help='Path to NPY file containing point data, shape [Frames, Points, Attrs]')
    parser.add_argument('--wav_path', type=str, required=True, help='Path to WAV file containing the sound corresponding to the animation.')
    parser.add_argument('--out_path', type=str, default=None, help='Path to output MOV file.')
    parser.add_argument('--fps', type=float, default=30.0)
    args = parser.parse_args()
    if args.out_path is None:
        args.out_path = args.wav_path.replace(os.path.splitext(args.wav_path)[-1], '.mov')
    render_3d_animation_with_audio(args)