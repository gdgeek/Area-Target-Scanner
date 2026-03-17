#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Result of UV unwrapping via xatlas
@interface XAtlasResult : NSObject
/// float3 x,y,z interleaved (count = vertexCount * 3)
@property (nonatomic, strong) NSArray<NSNumber *> *vertices;
/// float3 nx,ny,nz interleaved (count = vertexCount * 3)
@property (nonatomic, strong) NSArray<NSNumber *> *normals;
/// float2 u,v interleaved (count = vertexCount * 2)
@property (nonatomic, strong) NSArray<NSNumber *> *uvCoordinates;
/// uint32 i0,i1,i2 interleaved (count = faceCount * 3)
@property (nonatomic, strong) NSArray<NSNumber *> *faces;
/// uint32 per vertex — maps each output vertex to its original input vertex index
@property (nonatomic, strong) NSArray<NSNumber *> *originalVertexIndices;
@property (nonatomic, assign) int vertexCount;
@property (nonatomic, assign) int faceCount;
@end

@interface XAtlasBridge : NSObject

/// Perform UV unwrapping on the given mesh data.
/// @param vertices Interleaved float3 vertex positions (count = vertexCount * 3)
/// @param vertexCount Number of vertices
/// @param normals Interleaved float3 vertex normals (count = vertexCount * 3)
/// @param faces Interleaved uint32 triangle indices (count = faceCount * 3)
/// @param faceCount Number of triangles
/// @param error On failure, set to an NSError describing the problem
/// @return XAtlasResult on success, nil on failure
+ (nullable XAtlasResult *)unwrapWithVertices:(const float *)vertices
                                  vertexCount:(int)vertexCount
                                      normals:(const float *)normals
                                        faces:(const uint32_t *)faces
                                    faceCount:(int)faceCount
                                        error:(NSError *_Nullable *_Nullable)error;

@end

NS_ASSUME_NONNULL_END
