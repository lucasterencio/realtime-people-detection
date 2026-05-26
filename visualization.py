import cv2
import config

def draw_person(frame, bbox, name, confidence, dwell_time):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), config.BOX_COLOR, 2)

    text = f'{name} {confidence:.2f}'
    if dwell_time > 0:
        text += f' {dwell_time:.1f}s'

    (text_width, _), _ = cv2.getTextSize(
        text, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS
    )

    cv2.rectangle(
        frame,
        (x1, y1 - config.TEXT_BOX_HEIGHT),
        (x1 + text_width + config.TEXT_PADDING * 2, y1),
        config.TEXT_BG_COLOR,
        cv2.FILLED,
    )

    cv2.putText(
        frame, text,
        (x1 + config.TEXT_PADDING, y1 - config.TEXT_PADDING),
        config.FONT, config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS,
    )
