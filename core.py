import numpy as np

class Simulation():

    def __init__(self, simsize, num_frames, dt):

        self.simsize = simsize
        self.num_frames = num_frames
        self.dt = dt

        self.G = 1 # Gravitational constant
        self.soft = 0.5 # Softening parameter

    # Generate randomly distributed bodies over specified ranges
    def generate_bodies(self, num_bodies, mass_range, position_range, velocity_range):

        self.num_bodies = num_bodies
        self.mass_range = mass_range
        self.position_range = position_range
        self.velocity_range = velocity_range

        self.masses = np.random.uniform(self.mass_range[0], self.mass_range[1], self.num_bodies)
        self.positions = np.random.uniform(self.position_range[0], self.position_range[1], (self.num_bodies, 2))
        self.velocities = np.random.uniform(self.velocity_range[0], self.velocity_range[1], (self.num_bodies, 2))

    # central_mass is mass of large central object
    # num_bodies and ranges refer to smaller orbiting bodies
    def generate_circular_orbits(self, num_bodies, mass_range, radii_range, central_mass):

        self.num_bodies = num_bodies
        self.mass_range = mass_range
        self.radii_range = radii_range

        # Initialize position and velocity arrays
        self.velocities = np.zeros((self.num_bodies, 2))
        self.positions = np.zeros((self.num_bodies, 2))

        # Generate masses, angles, and radii of orbiting bodies
        self.masses = np.random.uniform(self.mass_range[0], self.mass_range[1], self.num_bodies)
        self.angles = np.random.uniform(0, 2 * np.pi, self.num_bodies)
        self.radii = np.random.uniform(radii_range[0], radii_range[1], self.num_bodies)

        # Generate positions of orbiting bodies
        for i in range(self.num_bodies):
            self.positions[i] = [self.radii[i] * np.cos(self.angles[i]), self.radii[i] * np.sin(self.angles[i])]
        
        # Determine orbital velocities of smaller bodies
        for i in range(self.num_bodies):
            angle = np.arctan2(self.positions[i][1], self.positions[i][0]) # Angle of position vector from x-axis
            radius = np.linalg.norm(self.positions[i]) # Orbital radius
            v_mag = np.sqrt(central_mass / radius) # Magnitude of orbital velocity
            vx = v_mag * (-np.sin(angle)) # x-component of orbital velocity
            vy = v_mag * np.cos(angle) # y-component of orbital velocity
            self.velocities[i] = np.array([vx, vy]) # Orbital velocity

        # Add large central body
        self.masses[0] = central_mass
        self.positions[0] = 0
        self.velocities[0] = 0

    # bodies should be an array specifying the number of bodies in each cluster
    # cluster_positions should be an array specifying center of each cluster
    # cluster_velocities should be an array specifying velocity of each cluster
    # central_masses should be an array specifying mass of central body of each cluster
    def generate_clusters(self, bodies, cluster_positions, cluster_velocities, mass_range, radii_range, central_masses):

        self.bodies = bodies # Array of bodies in each cluster
        self.num_bodies = np.sum(bodies) # Total number of simulation bodies
        self.cluster_positions = cluster_positions
        self.cluster_velocities = cluster_velocities
        self.mass_range = mass_range
        self.radii_range = radii_range
        self.central_masses = central_masses

        # Initialize position and velocity arrays
        self.velocities = np.zeros((self.num_bodies, 2))
        self.positions = np.zeros((self.num_bodies, 2))

        # Generate masses, angles, and radii of orbiting bodies
        self.masses = np.random.uniform(self.mass_range[0], self.mass_range[1], self.num_bodies)
        self.angles = np.random.uniform(0, 2 * np.pi, self.num_bodies)
        self.radii = np.random.uniform(radii_range[0], radii_range[1], self.num_bodies)

        # Calculate cluster offsets
        cluster_offsets = np.cumsum(self.bodies)

        # Generate clusters
        for cluster_index in range(len(self.bodies)):
            # Generate orbiting bodies in cluster
            for body in range(self.bodies[cluster_index]):
                if cluster_index == 0:
                    body_index = body
                else:
                    body_index = cluster_offsets[cluster_index - 1] + body
                
                # Calculate relative position in cluster
                relative_position = [self.radii[body_index] * np.cos(self.angles[body_index]), self.radii[body_index] * np.sin(self.angles[body_index])]
               
                # Calculate absolute position in simulation
                self.positions[body_index] = cluster_positions[cluster_index] + relative_position

                # Calculate velocities of orbiting body
                v_mag = np.sqrt(self.central_masses[cluster_index] / self.radii[body_index])
                vx = v_mag * (-np.sin(self.angles[body_index]))
                vy = v_mag * np.cos(self.angles[body_index])
                self.velocities[body_index] = np.array([vx, vy]) + self.cluster_velocities[cluster_index]
            
            # Add central mass
            self.masses[cluster_offsets[cluster_index] - 1] = central_masses[cluster_index]
            self.positions[cluster_offsets[cluster_index] - 1] = cluster_positions[cluster_index]
            self.velocities[cluster_offsets[cluster_index] - 1] = self.cluster_velocities[cluster_index]



    def calculate_forces(self):

        # Displacement tensor
        r = self.positions[:, None, :] - self.positions[None, :, :] # shape (N, N, 2)

        # Distance matrix
        r_norm = np.linalg.norm(r, axis = 2) # shape (N, N)
        np.fill_diagonal(r_norm, np.inf) # Avoid division by zero

        # Force tensor
        F = self.masses[:, None, None] * self.masses[None, :, None] * r / (r_norm[:, :, None]**2 + self.soft**2)**(3/2) # shape (N, N, 2)

        # Net foce matrix
        self.F_net = np.sum(F, axis = 0) # shape (N, 2)


    def update_positions(self):

        # Integrate velocities and positions using Euler's method
        self.velocities = self.velocities + (self.F_net / self.masses[:, None]) * self.dt # shape (N, 2)
        self.positions = self.positions + self.velocities * self.dt # shape (N, 2)

    def run_sim(self):

        # Simulation time steps
        time_steps = np.arange(0, self.num_frames * self.dt, self.dt)

        # Initialize tensor for storing position data
        self.data = np.zeros((len(time_steps), self.num_bodies, 2)) # shape (num_frames, N, 2)

        for time_index, time in enumerate(time_steps):

            self.calculate_forces()
            self.update_positions()

            self.data[time_index] = self.positions

    def save_animation(self, filename = 'simulation.mp4'):

        from multiprocessing import Pool
        import subprocess
        from multiprocessing import Pool
        import os

        # Ensure 'temp' directory exists
        os.makedirs('temp', exist_ok=True)

        fps = 30

        # Sizes and colors
        sizes = [100 * mass**(1/3) for mass in self.masses]
        colors = 'cyan'
        
        # Divide frames into batches
        batch_size = 100
        frame_batches = [range(i, min(i + batch_size, self.num_frames)) for i in range(0, self.num_frames, batch_size)]

        for batch_index, frame_batch in enumerate(frame_batches):
            print(f"Rendering batch {batch_index + 1}/{len(frame_batches)}...")

            # Prepare arguments for multiprocessing
            args = [
                (frame, self.data, self.simsize, self.num_bodies, sizes, colors)
                for frame in frame_batch
            ]

            # Render frames in parallel
            with Pool() as pool:
                pool.map(render_frame, args)

        # Combine frames into a video using FFmpeg
        print("Combining frames into video...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-r", str(fps),  # Set frame rate
            "-i", "temp/frame_%04d.png",  # Input frame pattern
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            filename,
        ]
        # Redirect stdout and stderr to suppress verbose output
        with open(os.devnull, 'w') as devnull:
            subprocess.run(ffmpeg_cmd, stdout=devnull, stderr=devnull)
        
        # Clean up frame images
        print("Cleaning up frames...")
        for frame in range(self.num_frames):
            frame_file = f"temp/frame_{frame:04d}.png"
            if os.path.exists(frame_file):
                os.remove(frame_file)

        # Remove the temp folder
        os.rmdir('temp')

        print(f"Animation saved to {filename}")
    
# worker function to render a single frame
def render_frame(args):
        import matplotlib.pyplot as plt

        frame, data, simsize, num_bodies, sizes, colors = args

        # Set up the figure
        fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi = 100)
        ax.set_xlim(simsize * -0.5, simsize * 0.5)
        ax.set_ylim(simsize * -0.5, simsize * 0.5)
        ax.set_title(f'{num_bodies} Body Gravity Simulation')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_facecolor("black")  # Set background color
        fig.patch.set_facecolor("lightgray")  # Outside the plot
        ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha = 1)

        # Plot bodies
        positions = data[frame]
        ax.scatter(positions[:, 0], positions[:, 1], s=sizes, c=colors)

        # Save the frame as an image
        frame_filename = f"temp/frame_{frame:04d}.png"
        plt.savefig(frame_filename, dpi=100)
        plt.close(fig)

        return frame_filename

