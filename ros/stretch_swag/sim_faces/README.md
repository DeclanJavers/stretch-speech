# Faces for the MuJoCo cutouts

For `mujoco.launch.py perception:=faces`. One photo per person in `config/world.json`, named after
them: `Jeff.jpg`, `Maria.jpg`, `Sam.jpg`, `Priya.jpg`, `Leo.jpg` (.jpg / .jpeg / .png).

- A front-facing photo with the whole face visible, like a passport photo or a selfie. Any size.
- Use photos of people who agreed to it (e.g. the team). They stay on your machine: this folder's
  images are ignored by git.
- No photo: that person gets a drawn face. YuNet does detect it, but all drawn faces look alike
  to the recognizer, so they get confused with each other.
