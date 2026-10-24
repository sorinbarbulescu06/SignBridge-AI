#TO FULLY REMAKE

class SignTranslator:
    """
    Reprezintă blocul [CODE FOR SIGN TRANSLATION] din Zona Albastră.
    Primește DOAR dicționarul `sign_input_template` și returnează textul tradus.
    """

    def __init__(self):
        self.last_translated_word = None

    def translate(self, sign_input_template):
        """
        Analizează coordonatele din template.
        Returnează un cuvânt doar atunci când se schimbă semnul (pentru a nu face spam în UI).
        """
        if sign_input_template["hands_detected"] == 0:
            self.last_translated_word = None
            return None

        # Luăm prima mână detectată din template
        first_hand = sign_input_template["hands"][0]
        lms = first_hand["landmarks"]
        hand_type = first_hand["type"]

        # Verificăm ce degete sunt ridicate comparând vârful degetului cu articulația de sub el
        # (În coordonatele imaginii, axa Y crește în jos, deci y_varf < y_baza înseamnă deget ridicat)
        fingers_up = [
            lms[8]["y"] < lms[6]["y"],    # Arătător (Index)
            lms[12]["y"] < lms[10]["y"],  # Mijlociu (Middle)
            lms[16]["y"] < lms[14]["y"],  # Inelar (Ring)
            lms[20]["y"] < lms[18]["y"],  # Degetul mic (Pinky)
        ]

        count_up = sum(fingers_up)

        # Reguli demonstrative pe baza datelor din template (până antrenăm modelul AI complet)
        detected_word = None
        if count_up == 4:
            detected_word = f"[SALUT / PALMĂ DESCHISĂ ({hand_type})]"
        elif count_up == 0:
            detected_word = f"[STOP / PUMN ({hand_type})]"
        elif fingers_up == [True, True, False, False]:
            detected_word = "[VICTORIE / PACE]"
        elif fingers_up == [True, False, False, False]:
            detected_word = "[UNU / ATENȚIE]"
        elif fingers_up == [True, False, False, True]:
            detected_word = "[ROCK / ILY]"

        # Returnăm cuvântul doar dacă e diferit de ultimul semn afișat
        if detected_word and detected_word != self.last_translated_word:
            self.last_translated_word = detected_word
            return detected_word

        return None