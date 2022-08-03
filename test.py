import streamlit as st
import numpy as np
from monai.transforms import LoadImaged, RandKSpaceSpikeNoised, RandAffined
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()


img_path1: str = os.getenv("img_path1", "None")
label_path: str = os.getenv("label_path", "None")


sample_dict = {
    "image": img_path1,
    "label": label_path,
}

loader_dict = LoadImaged(keys=["image", "label"])


@st.experimental_memo
def load_data(sample_dict):
    tr_spiker = RandKSpaceSpikeNoised(
        keys=[
            "image",
        ],
        prob=0.5,
        intensity_range=(10, 20),
    )
    trRandAffined = RandAffined(
        keys=["image", "label"], prob=1, shear_range=0.7, rotate_range=0.5
    )

    def transform_data(dd):

        dd = tr_spiker(dd)
        dd = trRandAffined(dd)
        return dd

    dd = loader_dict(sample_dict)
    dd = transform_data(dd)
    label = dd["label"]
    ct_array = dd["image"]
    return ct_array, label


ct_array, label = load_data(sample_dict)
slice_idx_max = np.argmax(
    label.sum(
        axis=(
            0,
            1,
        )
    )
)

target = (ct_array - np.min(ct_array)) / np.ptp(ct_array)
target = (target * 255).astype(np.uint8)

if st.button("RESET TRANSFORM"):
    # Clear values from *all* memoized functions:
    # i.e. clear values from both square and cube
    st.experimental_memo.clear()
col1, col2 = st.columns(2)
slice_idx = st.slider("z slice_idx?", 0, ct_array.shape[2], int(slice_idx_max))

with col1:
    slices_pil = Image.fromarray(target[:, :, slice_idx])
    st.image(slices_pil)

with col2:
    mask = label[:, :, slice_idx]

    mask = (np.stack([mask] * 3, -1) * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask)
    mask_pil.putalpha(64)

    slices_pil.paste(im=mask_pil, box=(0, 0), mask=mask_pil)
    st.image(slices_pil)

# shape = st.slider('How old are you?', 0, 130, 25)
st.write("current z index  =  ", slice_idx, "/")

# image = np.zeros(shape=(100,2,z))
st.write(
    "ct_array shape ",
    ct_array.shape,
)

#
# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# fig  # 👈 D
# Interactive Streamlit elements, like these sliders, return their value.
# This gives you an extremely simple interaction model.
# iterations = st.sidebar.slider("Level of detail", 2, 20, 10, 1)
# separation = st.sidebar.slider("Separation", 0.7, 2.0, 0.7885)

# # Non-interactive elements return a placeholder to their location
# # in the app. Here we're storing progress_bar to update it later.

# # These two elements will be filled in later, so we create a placeholder
# # for them using st.empty()
# frame_text = st.sidebar.empty()
# image = st.empty()

# m, n, s = 960, 640, 400
# x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
# y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))

# for frame_num, a in enumerate(np.linspace(0.0, 4 * np.pi, 100)):
#     # Here were setting value for these two elements.
#     frame_text.text("Frame %i/100" % (frame_num + 1))

#     # Performing some fractal wizardry.
#     c = separation * np.exp(1j * a)
#     Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
#     C = np.full((n, m), c)
#     M: Any = np.full((n, m), True, dtype=bool)
#     N = np.zeros((n, m))

#     for i in range(iterations):
#         Z[M] = Z[M] * Z[M] + C[M]
#         M[np.abs(Z) > 2] = False
#         N[M] = i

#     # Update the image placeholder by calling the image() function on it.
#     image.image(1.0 - (N / N.max()), use_column_width=True)

# # We clear elements by calling empty on them.
# frame_text.empty()

# # Streamlit widgets automatically run the script from top to bottom. Since
# # this button is not connected to any other logic, it just causes a plain
# # rerun.
