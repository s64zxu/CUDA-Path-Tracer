#pragma once

#include <vector>
#include "scene.h"

namespace pathtrace_wavefront {
	void InitDataContainer(GuiDataContainer* guiData);
	void PathtraceInit(Scene* scene);
	void PathtraceFree();
	void Pathtrace(uchar4* pbo, int frame, int iteration);
}