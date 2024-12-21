from rvc_inferpy.infer import infer_audio


class infernew:
    def __init__(
        self,
        model_name,
        sound_path,
        f0_change=0,
        f0_method="rmvpe",
        min_pitch=50,
        max_pitch=800,
        crepe_hop_length=128,
        index_rate=1.0,
        filter_radius=3,
        rms_mix_rate=0.75,
        protect=0.33,
        split_infer=True,
        min_silence=0.5,
        silence_threshold=-40,
        seek_step=10,
        keep_silence=0.1,
        quefrency=0.0,
        timbre=1.0,
        f0_autotune=False,
        output_format="wav",
    ):
        self.model_name = model_name
        self.sound_path = sound_path
        self.f0_change = f0_change
        self.f0_method = f0_method
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.crepe_hop_length = crepe_hop_length
        self.index_rate = index_rate
        self.filter_radius = filter_radius
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect
        self.split_infer = split_infer
        self.min_silence = min_silence
        self.silence_threshold = silence_threshold
        self.seek_step = seek_step
        self.keep_silence = keep_silence
        self.quefrency = quefrency
        self.timbre = timbre
        self.f0_autotune = f0_autotune
        self.output_format = output_format

    def run_inference(self):
        inferred_audio = infer_audio(
            MODEL_NAME=self.model_name,
            SOUND_PATH=self.sound_path,
            F0_CHANGE=self.f0_change,
            F0_METHOD=self.f0_method,
            MIN_PITCH=self.min_pitch,
            MAX_PITCH=self.max_pitch,
            CREPE_HOP_LENGTH=self.crepe_hop_length,
            INDEX_RATE=self.index_rate,
            FILTER_RADIUS=self.filter_radius,
            RMS_MIX_RATE=self.rms_mix_rate,
            PROTECT=self.protect,
            SPLIT_INFER=self.split_infer,
            MIN_SILENCE=self.min_silence,
            SILENCE_THRESHOLD=self.silence_threshold,
            SEEK_STEP=self.seek_step,
            KEEP_SILENCE=self.keep_silence,
            QUEFRENCY=self.quefrency,
            TIMBRE=self.timbre,
            F0_AUTOTUNE=self.f0_autotune,
            OUTPUT_FORMAT=self.output_format,
        )
        return inferred_audio
