"""Live2D Cubism 3.0/5.0 loader via Core C API (ctypes).  v4.5.0 §7.3"""

import ctypes, json, mmap, os, sys, logging

logger = logging.getLogger(__name__)

# Core library location — copied from SDK at first use
_CORE_LIB = os.path.join(os.path.dirname(__file__), "libs", "libLive2DCubismCore.so")
ALIGN_MOC = 64
ALIGN_MODEL = 16


class csmVector2(ctypes.Structure):
    """Live2D Cubism Core 2D vector for vertex positions / UVs.  v4.5.0 §7.3"""
    _fields_ = [("X", ctypes.c_float), ("Y", ctypes.c_float)]


def _alloc_aligned(size: int, align: int):
    """Allocate 64/16-byte aligned buffer via mmap. Returns (buffer, mmap_ref)."""
    b = mmap.mmap(-1, size + align, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    a = ctypes.addressof(ctypes.c_char.from_buffer(b))
    off = (align - (a % align)) % align
    buf = (ctypes.c_char * (size + align)).from_buffer(b, off)
    return buf, b


class Live2DModel:
    """Loads and queries a Live2D Cubism 3.0/5.0 MOC3 model."""

    def __init__(self, core_path: str = None):
        path = core_path or _CORE_LIB
        if not os.path.exists(path):
            raise FileNotFoundError(f"Core library not found: {path}")
        self._lib = ctypes.CDLL(path)
        self._moc = None
        self._model = None
        self._refs = []  # keep mmap refs alive

    # ------------------------------------------------------------------ load
    def load(self, model3_json: str) -> bool:
        base = os.path.dirname(model3_json)
        spec = json.load(open(model3_json))
        moc_file = os.path.join(base, spec["FileReferences"]["Moc"])
        logger.info(f"Loading MOC3: {moc_file}")
        raw = open(moc_file, "rb").read()
        logger.debug(f"MOC3 size: {len(raw)} B")

        # --- revive MOC (64-byte aligned) ---
        buf, ref = _alloc_aligned(len(raw), ALIGN_MOC); self._refs.append(ref)
        ctypes.memmove(buf, raw, len(raw))

        self._set_api("csmReviveMocInPlace", [ctypes.c_void_p, ctypes.c_uint], ctypes.c_void_p)
        self._moc = self._lib.csmReviveMocInPlace(buf, len(raw))
        if not self._moc:
            raise RuntimeError("csmReviveMocInPlace returned NULL — invalid or corrupted MOC3")

        # --- init model (16-byte aligned) ---
        self._set_api("csmGetSizeofModel", [ctypes.c_void_p], ctypes.c_uint)
        ms = self._lib.csmGetSizeofModel(self._moc)
        logger.debug(f"Model memory: {ms} B")

        mbuf, mref = _alloc_aligned(ms, ALIGN_MODEL); self._refs.append(mref)
        # CORRECT signature: (moc, address, size)
        self._set_api("csmInitializeModelInPlace",
                      [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint], ctypes.c_void_p)
        self._model = self._lib.csmInitializeModelInPlace(self._moc, mbuf, ms)
        if not self._model:
            raise RuntimeError("csmInitializeModelInPlace returned NULL")

        # --- drawable vertex/index API bindings  v4.5.0 §7.3 ---
        self._set_api("csmGetDrawableVertexCounts", [ctypes.c_void_p], ctypes.POINTER(ctypes.c_int))
        self._set_api("csmGetDrawableVertexPositions", [ctypes.c_void_p], ctypes.POINTER(ctypes.POINTER(csmVector2)))
        self._set_api("csmGetDrawableVertexUvs", [ctypes.c_void_p], ctypes.POINTER(ctypes.POINTER(csmVector2)))
        self._set_api("csmGetDrawableIndexCounts", [ctypes.c_void_p], ctypes.POINTER(ctypes.c_int))
        self._set_api("csmGetDrawableIndices", [ctypes.c_void_p], ctypes.POINTER(ctypes.POINTER(ctypes.c_ushort)))
        self._set_api("csmGetDrawableConstantFlags", [ctypes.c_void_p], ctypes.POINTER(ctypes.c_ubyte))
        self._set_api("csmGetDrawableOpacities", [ctypes.c_void_p], ctypes.POINTER(ctypes.c_float))
        self._set_api("csmGetDrawableRenderOrders", [ctypes.c_void_p], ctypes.POINTER(ctypes.c_int))
        self._set_api("csmGetDrawableTextureIndices", [ctypes.c_void_p], ctypes.POINTER(ctypes.c_int))
        self._set_api("csmGetDrawableMaskCounts", [ctypes.c_void_p], ctypes.POINTER(ctypes.c_int))
        self._set_api("csmGetDrawableMasks", [ctypes.c_void_p], ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))

        logger.info(f"Model loaded: {self.param_count} params, {self.part_count} parts, "
                     f"{self.drawable_count} drawables")
        return True

    def _set_api(self, name, argtypes, restype):
        fn = getattr(self._lib, name)
        fn.argtypes = argtypes
        fn.restype = restype

    # ------------------------------------------------------------------ queries
    @property
    def param_count(self):
        self._set_api("csmGetParameterCount", [ctypes.c_void_p], ctypes.c_int)
        return self._lib.csmGetParameterCount(self._model)

    @property
    def part_count(self):
        self._set_api("csmGetPartCount", [ctypes.c_void_p], ctypes.c_int)
        return self._lib.csmGetPartCount(self._model)

    @property
    def drawable_count(self):
        self._set_api("csmGetDrawableCount", [ctypes.c_void_p], ctypes.c_int)
        return self._lib.csmGetDrawableCount(self._model)

    # --- drawable vertex/index data  v4.5.0 §7.3 ---
    def vertex_counts(self) -> list[int]:
        ptr = self._lib.csmGetDrawableVertexCounts(self._model)
        return [ptr[i] for i in range(self.drawable_count)]

    def vertex_positions(self) -> list[list[tuple[float, float]]]:
        ptr = self._lib.csmGetDrawableVertexPositions(self._model)
        counts = self.vertex_counts()
        result = []
        for i, n in enumerate(counts):
            arr = ptr[i]
            result.append([(arr[j].X, arr[j].Y) for j in range(n)])
        return result

    def vertex_uvs(self) -> list[list[tuple[float, float]]]:
        ptr = self._lib.csmGetDrawableVertexUvs(self._model)
        counts = self.vertex_counts()
        result = []
        for i, n in enumerate(counts):
            arr = ptr[i]
            result.append([(arr[j].X, arr[j].Y) for j in range(n)])
        return result

    def index_counts(self) -> list[int]:
        ptr = self._lib.csmGetDrawableIndexCounts(self._model)
        return [ptr[i] for i in range(self.drawable_count)]

    def indices(self) -> list[list[int]]:
        ptr = self._lib.csmGetDrawableIndices(self._model)
        cnts = self.index_counts()
        result = []
        for i, n in enumerate(cnts):
            arr = ptr[i]
            result.append([arr[j] for j in range(n)])
        return result

    def constant_flags(self) -> list[int]:
        ptr = self._lib.csmGetDrawableConstantFlags(self._model)
        return [ptr[i] for i in range(self.drawable_count)]

    def opacities(self) -> list[float]:
        ptr = self._lib.csmGetDrawableOpacities(self._model)
        return [ptr[i] for i in range(self.drawable_count)]

    def render_orders(self) -> list[int]:
        ptr = self._lib.csmGetDrawableRenderOrders(self._model)
        return [ptr[i] for i in range(self.drawable_count)]

    def texture_indices(self) -> list[int]:
        ptr = self._lib.csmGetDrawableTextureIndices(self._model)
        return [ptr[i] for i in range(self.drawable_count)]

    def mask_counts(self) -> list[int]:
        ptr = self._lib.csmGetDrawableMaskCounts(self._model)
        return [ptr[i] for i in range(self.drawable_count)]

    def masks(self) -> list[list[int]]:
        ptr = self._lib.csmGetDrawableMasks(self._model)
        cnts = self.mask_counts()
        result = []
        for i, n in enumerate(cnts):
            if n == 0:
                result.append([])
            else:
                arr = ctypes.cast(ptr[i], ctypes.POINTER(ctypes.c_int))
                result.append([arr[j] for j in range(n)])
        return result

    # ------------------------------------------------------------------ parameter manipulation
    def set_parameter(self, param_id: str, value: float) -> bool:
        """Set a Live2D parameter value by its string ID.  v4.5.0 §7.3"""
        # Resolve parameter ID → index via csmGetParameterIds
        self._set_api("csmGetParameterIds", [ctypes.c_void_p], ctypes.c_void_p)
        ids_addr = self._lib.csmGetParameterIds(self._model)
        if not ids_addr:
            logger.warning("Live2DModel.set_parameter: csmGetParameterIds returned NULL")
            return False
        ids_ptr = ctypes.cast(ids_addr, ctypes.POINTER(ctypes.c_char_p))

        n = self.param_count
        target_idx = -1
        for i in range(n):
            c_str = ids_ptr[i]
            if c_str is None:
                continue
            name = c_str.decode("utf-8") if isinstance(c_str, bytes) else c_str
            if name == param_id:
                target_idx = i
                break

        if target_idx < 0:
            logger.debug("Live2DModel.set_parameter: param_id=%r not found (count=%d)", param_id, n)
            return False

        # Write value into the model's writable parameter array
        self._set_api("csmGetParameterValues", [ctypes.c_void_p], ctypes.c_void_p)
        vals_addr = self._lib.csmGetParameterValues(self._model)
        if not vals_addr:
            logger.warning("Live2DModel.set_parameter: csmGetParameterValues returned NULL")
            return False
        vals_ptr = ctypes.cast(vals_addr, ctypes.POINTER(ctypes.c_float))
        vals_ptr[target_idx] = value
        return True

    def start_motion(self, motion_name: str) -> bool:
        """Start a Live2D motion (stub — Cubism Core does not handle motions).  v4.5.0 §7.3"""
        logger.info("Live2DModel.start_motion: %s (stub — requires CubismFramework layer)", motion_name)
        self._current_motion = motion_name
        return True

    # ------------------------------------------------------------------ lifecycle
    def update(self):
        self._set_api("csmUpdateModel", [ctypes.c_void_p], None)
        self._lib.csmUpdateModel(self._model)

    def close(self):
        self._moc = None
        self._model = None
        # Keep self._refs alive — Cubism Core atexit handler needs mmap buffers
        # (Python GC may otherwise free them before Cubism's atexit runs)


# ================================================================ subprocess mode
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model3.json>", file=sys.stderr)
        sys.exit(1)
    m = Live2DModel()
    m.load(sys.argv[1])
    print(f"params={m.param_count} parts={m.part_count} drawables={m.drawable_count}")
    m.close()
