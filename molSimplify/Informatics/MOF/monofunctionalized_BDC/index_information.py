# Each entry corresponds to a different functional group.
# Columns: functional group name, anchor functional group atom index, functional group atom indices, base carbon index, neighbor carbon indices
# Zero-indexing.
# Add new entries below along with a corresponding XYZ file in this folder. XYZ file should have the same name as the functional group.
INDEX_INFO = {
	"OCF3": [6, [6,18,19,20,21], 0, [1,5]],
	"SO3H": [6, [6,18,19,20,21], 0, [1,5]],
	"OCH3": [11, [11,18,19,20,21], 5, [0,4]],
	"CO2H": [6, [6,18,19,20], 0, [1,5]],
}
