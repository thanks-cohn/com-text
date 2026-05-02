   ██████╗ ██████╗ ███╗   ███╗      ████████╗███████╗██╗  ██╗████████╗
  ██╔════╝██╔═══██╗████╗ ████║      ╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝
  ██║     ██║   ██║██╔████╔██║█████╗   ██║   █████╗   ╚███╔╝    ██║   
  ██║     ██║   ██║██║╚██╔╝██║╚════╝   ██║   ██╔══╝   ██╔██╗    ██║   
  ╚██████╗╚██████╔╝██║ ╚═╝ ██║          ██║   ███████╗██╔╝ ██╗   ██║   
   ╚═════╝ ╚═════╝ ╚═╝     ╚═╝          ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   

                 com-text :: spatial ocr → svg engine


  what this is

  com-text takes a page — pdf, image, or a folder — and extracts text line by line,
  preserving exactly where each line exists on the page.

  instead of flattening everything into a block of text, it rebuilds it as structured
  svg text objects layered over the original image.

  your comic is no longer just an image — it becomes something you can inspect,
  search, and reshape.


  quick start

    python com-text.py chapter.pdf --out out_dir

    python com-text.py chapter.pdf --max-pages 3
    python com-text.py chapter.pdf --show-boxes

  simple: point it at something, get structured text back.


  what goes in

    pdf files
    images (png, jpg, webp, tiff, bmp)
    directories of images

  everything is normalized into page images internally.


  what comes out

    out/
    ├── svg/      → pages with positioned text objects
    ├── json/     → structured text data
    ├── _work/    → rendered page images
    └── manifest.json

  each json file contains:

    page number
    image path
    width / height
    list of text lines:
      text
      bounding box
      confidence
      polygon (when available)

  each svg contains:

    original page image
    +
    <text> elements placed exactly where text was found


  why this matters

  most OCR systems flatten everything.

  com-text keeps structure:

    dialogue stays where it was
    captions stay where they were
    layout remains intact

  this opens the door to far more than extraction.


  use cases

  for comic / manga enthusiasts:

    build searchable manga archives
    extract dialogue from favorite scenes instantly
    create translation overlays
    experiment with re-typesetting pages
    explore how dialogue flows across panels

  for researchers:

    analyze spatial distribution of text
    study reading flow and layout patterns
    generate structured datasets
    align text with panel systems
    measure density and pacing

  for builders:

    feed json into searchable databases (elasticsearch, sqlite, etc.)
    build comic search engines ("find every time this character speaks")
    create svg-native comic readers
    layer annotations or translations
    plug into multimodal AI pipelines

  everything needed is already in the output:

    text + position + page structure + consistent ids


  flags

    --out               output directory
    --dpi               pdf render quality
    --first-page        start page
    --max-pages         limit pages
    --recursive         scan directories
    --lang              ocr language
    --min-confidence    filter weak detections
    --show-boxes        draw bounding boxes
    --hide-text         keep objects but hide visible text
    --no-embed-image    link images instead of embedding
    --save-crops        export text-line images
    --debug-ocr         inspect raw ocr output


  installation

    pip install pymupdf pillow paddleocr paddlepaddle

  if paddle causes issues:

    pip uninstall -y paddlepaddle
    pip install paddlepaddle==3.2.2


  next evolution

  currently:

    image + svg text overlay

  next step:

    remove the underlying raster text
    keep only svg text objects

  this enables:

    cleaner and more aesthetic outputs
    full typography control
    resolution-independent pages
    editable dialogue layers
    true text-native comics

  not just extraction — reconstruction.


  philosophy

  a line of text is not just text

    it is a position
    it is a region
    it is part of the visual rhythm

  com-text preserves that.


SHOUTOUT TO NEIL COHN! thanks for inspiring me to actually finish this project!
Best of wishes!



  and once you see your comics this way
  it is hard to go back
