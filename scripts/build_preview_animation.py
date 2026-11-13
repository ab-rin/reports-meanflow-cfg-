"""Render the README animation from recorded model states; no interpolation."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
font = ImageFont.truetype(str(ASSETS / 'fonts/IBMPlexMono-Light.ttf'), 14)
title = ImageFont.truetype(str(ASSETS / 'fonts/EBGaramond.ttf'), 40)
frames = []
names = ['pikachu', 'bulbasaur', 'charmander', 'squirtle', 'eevee', 'gengar']
for step in range(5):
    canvas = Image.new('RGB', (1040, 320), '#fafbf9')
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 22), 'Pokémon from noise.', font=title, fill='#3c443c')
    label = 'initial noise' if step == 0 else f'call {step} / 4'
    draw.text((1008, 47), label, font=font, fill='#47675b', anchor='ra')
    for index, name in enumerate(names):
        strip = Image.open(ASSETS / 'gallery' / f'{name}-42-steps.png')
        state = strip.crop((step * 64, 0, (step + 1) * 64, 64))
        canvas.paste(state.resize((144, 144), Image.Resampling.NEAREST), (30 + 166 * index, 101))
        draw.text((30 + 166 * index, 258), name, font=font, fill='#667061')
    draw.text((30, 295), '101k EMA · seed 42 · actual recorded states · [1, .6, .3, .018, 0]', font=font, fill='#667061')
    frames.append(canvas)
frames[0].save(ASSETS / 'four-calls.gif', save_all=True, append_images=frames[1:], duration=[900, 700, 700, 700, 2300], loop=0, optimize=True, disposal=2)
print('Wrote five recorded frames to docs/assets/four-calls.gif')
