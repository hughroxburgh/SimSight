import numpy as np
from scipy.spatial import cKDTree
import sys

TOLERANCE = 1E-9
SIZE_MAX = sys.maxsize

def _Check_Mode(tree,t,current_tree_idx, next_tree_idx):
        """
        Finds the closest point to the location where the intersecting plane of two Voronoi cells meets the z-axis/ray-axis.
            1. Checks the closest two points.
            2. If a new point (not current or next idx) is clearly closer, exit with mode = 3
            3. If both current and next idx are in the top 8 closest within tolerance, exit with mode = 0, meaning finished.
            4. If somehow only one of the current and next is in the top 8 closest, a very weird boundary has been reached and something has gone wrong.   
        """

        # -- initialise mode at undetermined -- #
        mode = 3

        # -- search for the (k) two nearest points to the intersection point on the ray ([0,0,t]) -- #
        k = 2
        distances, indices = tree.query(np.array([0,0,t]), k=k)  

        # -- check if closest point is either of the known points -- #
        if indices[0] == current_tree_idx:
            mode -= 2
        if indices[0] == next_tree_idx:
            mode -= 1

        # -- Iterate through 8 closest points but break as soon as a point isnt close within the tolerance -- #
        i = 1
        while i < 8:
            if distances[i] - distances[0] <= TOLERANCE:    # check if distance to [0,0,t] is also very small
                
                # check if closest point is either of the known points # 
                if indices[i] == current_tree_idx:
                    mode -= 2
                if indices[i] == next_tree_idx:
                    mode -= 1

                # if both current and next in closest 8, we're good 
                if mode == 0:
                    break

                i += 1
                distances, indices = tree.query(np.array([0,0,t]), k=i + 1)

            else:   
                break

        # some failure has occurred 
        if mode in [1, 2]:
            print(f"mode={mode}, ctree_id={current_tree_idx}, ntree_id={next_tree_idx}")
            print(f"r2={distances[0:4]}")
            print(f"r2-r2[0]={distances[0:4] - distances[0]}")
            print(f"result={indices[0:4]}")

        return indices[0], mode

class RayPoint:
    """Represents a point along the ray where the ray enters/exits a Voronoi cell."""
    def __init__(self, tree_id, next_index, t):
        self.tree_id = tree_id  # Voronoi cell index this point belongs to
        self.next = next_index  # Index of the next RayPoint in the linked list
        self.t = t              # Distance along the ray

class RaySegment:
    """Represents a segment of the ray that lies entirely within a single Voronoi cell."""
    def __init__(self, tree_id, t1, t2, dt):
        self.tree_id = tree_id  # Voronoi cell index
        self.t1 = t1            # Start distance along ray
        self.t2 = t2            # End distance along ray
        self.dt = dt            # Segment length (s2 - s1)

class Ray:
    """Represents a ray in 3D space with methods to compute its intersections with a Voronoi cloud."""
    def __init__(self, length):

        self.length = length
        self.direction_vector = np.array([0,0,1],dtype=np.float32)

        # Initialize points along the ray (start and end)
        self.pts = []
        self.pts.append(RayPoint(SIZE_MAX, 1, 0.0))         # Start point
        self.pts.append(RayPoint(SIZE_MAX, SIZE_MAX, self.length))  # End point

        # Initialize empty list for segments
        self.segments = []


    def find_split_point_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute the distance along the ray where it splits between two Voronoi points.
        This uses a projection method based on the ray direction.
        
            ray: td  &  bisecting plane: n.(x-m)=0          (m can be any point on plane, here it is midpoint)
                -> set x = ray = td,   n.(td-m)=0
                -> therefore, t(n.d)-(n.m)=0
                -> t = (n.m)/(n.d)
        """
        normal_vector = point2 - point1     # normal vector to bisecting plane
        midpoint = 0.5 * (point1 + point2)  # midpoint between points, lies on bisecting plane

        norm_dot_midpoint = np.dot(normal_vector, midpoint)         
        norm_dot_ray = np.dot(normal_vector, self.direction_vector)

        return norm_dot_midpoint / norm_dot_ray


    def compute_ray_segments(self, tree, verbose=False):
        """
        Compute the segments of the ray that traverse different Voronoi cells.
        Stores results in self.segments.
            1. Starts by checking point A->B. It finds point C in between.
            2. Now it checks A->C. It finds nothing in between, so adds the segment A->t and t->C
            3. Now it checks C->B. It finds nothing in between, so adds the segment C->t and t->B. 
        """

        self.segments.clear()   # clear segments list

        # Query the Voronoi tree for the origin and endpoint and add to point list
        self.pts[0].tree_id = tree.query(np.array([0,0,0]))[1]
        self.pts[1].tree_id = tree.query(np.array([0,0,self.length]))[1]

        # If the entire ray is inside one cell, create a single segment
        if self.pts[0].tree_id == self.pts[1].tree_id:
            self.segments.append(RaySegment(
                self.pts[0].tree_id, 0, self.length, self.length))
            return

        # Traverse the linked list of points along the ray
        current_idx = 0
        next_idx = 1
        current_tree_id = self.pts[current_idx].tree_id
        next_tree_id = self.pts[next_idx].tree_id

        not_done = True
        count = 0
        while not_done:
            if verbose:
                print(count)

            # Compute split distance along the ray between the two cell points
            t = self.find_split_point_distance(tree.data[current_tree_id], tree.data[next_tree_id])

            # Check which cell the split belongs to
            closest_tree_id, mode = _Check_Mode(tree, t, current_tree_id, next_tree_id)

            # Ray transitions normally between two cells; create segments
            if mode == 0:
                dt = t - self.pts[current_idx].t
                self.segments.append(RaySegment(current_tree_id, self.pts[current_idx].t, t, dt))

                dt = self.pts[next_idx].t - t
                self.segments.append(RaySegment(next_tree_id, t, self.pts[next_idx].t, dt))

                # Move to the next point along the ray
                current_idx = self.pts[current_idx].next
                current_tree_id = self.pts[current_idx].tree_id
                next_idx = self.pts[current_idx].next

                # If there is no next_idx, we have finished 
                if next_idx == SIZE_MAX:
                    not_done = False
                else:
                    next_tree_id = self.pts[next_idx].tree_id

            # A particle has been found between, so we insert a new split point along the ray
            elif mode == 3:
                self.pts.append(RayPoint(closest_tree_id, next_idx, t)) # add new point to end of array that retains original next idx
                new_idx = len(self.pts) - 1  

                self.pts[current_idx].next = new_idx    # change next_idx of current point
                
                # change idx of "next" cell checking
                next_idx = new_idx  
                next_tree_id = closest_tree_id  
            else:
                raise ValueError('ModeError')

            count += 1


    def extract_line_elements(self):
        """
        Consolidates consecutive segments belonging to the same Voronoi cell.
        Returns: np.array of [[tree_id, total_segment_length], ...]
        """

        ids = []
        lengths = []
        
        # Start with the first segment
        current_id = self.segments[0].tree_id
        current_length = self.segments[0].dt 

        for i in range(1, len(self.segments)):
            seg = self.segments[i]
            
            if seg.tree_id == current_id:
                # If it's the same cell, just add to the length
                current_length += seg.dt
            else:
                # If it's a new cell, push the finished one and start a new count
                ids.append(current_id)
                lengths.append(current_length)

                current_id = seg.tree_id
                current_length = seg.dt

        # Don't forget to push the last cell
        ids.append(current_id)
        lengths.append(current_length)

        return np.array(ids),np.array(lengths)


def Find_Line_Elements(points,length):
    """
    Returns an array of [idx,dt]
    """

    tree = cKDTree(points) 

    ray = Ray(length)   
    ray.compute_ray_segments(tree) 
    ids,lengths = ray.extract_line_elements()

    return ids,lengths.astype(np.float32)






















































    # def integrate(self, cloud):
    #     self.segments.clear()
    #     self.dens_col = 0.0

    #     self.pts[0].tree_id = cloud.query_tree(self.pos_start)
    #     self.pts[1].tree_id = cloud.query_tree(self.pos_end)

    #     if self.pts[0].tree_id == self.pts[1].tree_id:
    #         raise RuntimeError(
    #             f"Start and end point are in the same cell. "
    #             f"Start point tree_id: {self.pts[0].tree_id}, End point tree_id: {self.pts[1].tree_id}"
    #         )

    #     current = 0
    #     next_idx = 1
    #     ctree_id = self.pts[current].tree_id
    #     ntree_id = self.pts[next_idx].tree_id
    #     not_done = True

    #     while not_done:
    #         s = self.find_split_point_distance(cloud.get_pt(ctree_id), cloud.get_pt(ntree_id))
    #         pos = self.pos_start + s * self.dir

    #         stree_id, mode = cloud.check_mode(pos, ctree_id, ntree_id)

    #         if mode == 0:
    #             ds = s - self.pts[current].s
    #             self.dens_col += ds * cloud.get_dens(ctree_id)
    #             self.segments.append(RaySegment(ctree_id, self.pts[current].s, s, ds))

    #             ds = self.pts[next_idx].s - s
    #             self.dens_col += ds * cloud.get_dens(ntree_id)
    #             self.segments.append(RaySegment(ntree_id, s, self.pts[next_idx].s, ds))

    #             current = self.pts[current].next
    #             ctree_id = self.pts[current].tree_id
    #             next_idx = self.pts[current].next

    #             if next_idx == SIZE_MAX:
    #                 not_done = False
    #             else:
    #                 ntree_id = self.pts[next_idx].tree_id

    #         elif mode == 1:
    #             print("Unlucky! mode=1")
    #             raise RuntimeError("Invalid edge condition (mode=1)")
    #         elif mode == 2:
    #             print("Unlucky! mode=2")
    #             raise RuntimeError("Invalid edge condition (mode=2)")
    #         elif mode == 3:
    #             self.pts.append(RayPoint(stree_id, next_idx, s))
    #             new_idx = len(self.pts) - 1
    #             self.pts[current].next = new_idx
    #             next_idx = new_idx
    #             ntree_id = stree_id

    # def get_dens_col(self):
    #     return self.dens_col
