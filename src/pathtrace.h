#pragma once

#include <vector>
#include "scene.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>


namespace pathtrace{
	void InitDataContainer(GuiDataContainer* guiData);
	void PathtraceInit(Scene* scene);
	void PathtraceFree();
	void Pathtrace(uchar4* pbo, int frame, int iteration);
}
