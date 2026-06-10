from scipy.ndimage import label
from scipy.ndimage import center_of_mass
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score as sk_auc


class Model_actions:
    def __init__(self, model):
        self.model = model
        self.all_patients = []
        self.dice = None
        self.model_auc = None
        self.pr_auc = None
        self.precision = None
        self.recall = None
        self.lw_precision = None
        self.lw_recall = None
        self.mean_center_distance = None
        self.median_center_distance = None

    # ======================================================================
    # Training / evaluation entry-points
    # ======================================================================

    def train(self, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):

        train_losses, train_dices, train_focals = [], [], []
        val_losses,   val_dices,   val_focals   = [], [], []

        self.model.to(device)

        for epoch in range(num_epochs):
            self.model.train()

            running_total_loss = 0.0
            running_dice_loss  = 0.0
            running_focal_loss = 0.0

            # FIX: Dataset_combined returns (images, masks, idx) — three values.
            #      Dataset_v1 also now returns three values.  Using *_ makes the
            #      loop safe for any dataset regardless of extra return values.
            for images, masks, *_ in train_loader:
                images = images.to(device)
                masks  = masks.float().to(device)

                optimizer.zero_grad()

                outputs = self.model(images)
                loss    = self.combined_loss(outputs, masks)

                loss.backward()
                optimizer.step()

                n = images.size(0)
                running_total_loss += loss.item() * n

                # FIX: wrap in no_grad — after backward() the graph for
                #      `outputs` is freed; recomputing dice/focal without
                #      no_grad would build a new (unused) graph, wasting memory.
                with torch.no_grad():
                    running_dice_loss  += self.dice_loss(outputs, masks).item() * n
                    running_focal_loss += self.focal_loss(outputs, masks).item() * n

            epoch_loss  = running_total_loss / len(train_loader.dataset)
            epoch_dice  = running_dice_loss  / len(train_loader.dataset)
            epoch_focal = running_focal_loss / len(train_loader.dataset)

            train_losses.append(epoch_loss)
            train_dices.append(epoch_dice)
            train_focals.append(epoch_focal)

            # ------------------------------------------------------------------
            # Validation phase
            # ------------------------------------------------------------------
            self.model.eval()

            val_running_total_loss = 0.0
            val_running_dice_loss  = 0.0
            val_running_focal_loss = 0.0

            with torch.no_grad():
                for images, masks, *_ in val_loader:   # FIX: same *_ unpack
                    images = images.to(device)
                    masks  = masks.float().to(device)

                    outputs = self.model(images)
                    loss    = self.combined_loss(outputs, masks)

                    n = images.size(0)
                    val_running_total_loss += loss.item() * n
                    val_running_dice_loss  += self.dice_loss(outputs, masks).item() * n
                    val_running_focal_loss += self.focal_loss(outputs, masks).item() * n

            val_loss  = val_running_total_loss / len(val_loader.dataset)
            val_dice  = val_running_dice_loss  / len(val_loader.dataset)
            val_focal = val_running_focal_loss / len(val_loader.dataset)

            val_losses.append(val_loss)
            val_dices.append(val_dice)
            val_focals.append(val_focal)

            if scheduler is not None:
                scheduler.step()

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, Focal: {epoch_focal:.4f} | "
                f"Val   Loss: {val_loss:.4f},   Dice: {val_dice:.4f},   Focal: {val_focal:.4f}"
            )

        return train_losses, val_losses, train_dices, val_dices, train_focals, val_focals

    # ------------------------------------------------------------------

    def evaluate(self, dataloader, device, threshold=0.5, show=False):
        '''
        Evaluates the model on:

        Voxel-wise  : Dice, Precision, Recall
        Lesion-wise : Precision, Recall, Mean/Median centroid distance
        Heatmap     : PR-AUC, ROC-AUC
        '''
        all_patients = self.predict(dataloader, device, threshold)

        # -- Voxel-wise -------------------------------------------------------
        all_labels = np.concatenate([p._labels.flatten() for p in all_patients])
        all_preds  = np.concatenate([p._preds.flatten()  for p in all_patients])
        all_probs  = np.concatenate([p._probs.flatten()  for p in all_patients])

        dice = self.dice_score(all_preds, all_labels)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels,   all_preds, zero_division=0)

        # -- Lesion-wise -------------------------------------------------------
        # FIX: merged the two separate patient loops into one so assign_lesions()
        #      is called once (via find_TP_FP_FN) and center_distance() reuses
        #      the cached result via the _lesions_assigned flag.
        model_TP, model_FP, model_FN = 0, 0, 0
        center_distances = []

        for patient in tqdm(all_patients, desc='Lesion-wise metrics'):
            patient.find_TP_FP_FN()
            model_TP += patient.TP
            model_FP += patient.FP
            model_FN += patient.FN
            center_distances.extend(patient.center_distance())  # reuses cached assignment

        lw_precision = model_TP / (model_TP + model_FP) if (model_TP + model_FP) > 0 else float('nan')
        lw_recall    = model_TP / (model_TP + model_FN) if (model_TP + model_FN) > 0 else float('nan')

        mean_center_distance   = float(np.mean(center_distances))   if center_distances else float('nan')
        median_center_distance = float(np.median(center_distances)) if center_distances else float('nan')

        # -- Heatmap quality --------------------------------------------------
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc    = auc(recall_curve, precision_curve)
        model_auc = sk_auc(all_labels, all_probs)

        # Store on self
        self.dice                  = dice
        self.model_auc             = model_auc
        self.pr_auc                = pr_auc
        self.precision             = prec
        self.recall                = rec
        self.lw_precision          = lw_precision
        self.lw_recall             = lw_recall
        self.mean_center_distance  = mean_center_distance
        self.median_center_distance = median_center_distance

        if show:
            self.show_evaluation()

        # FIX: trailing comma was turning the return into a 10-tuple with a
        #      spurious None at the end.
        return dice, model_auc, pr_auc, prec, rec, lw_precision, lw_recall, mean_center_distance, median_center_distance

    # ------------------------------------------------------------------

    def predict(self, dataloader, device, threshold=0.7):
        self.model.eval()
        all_patients = []

        with torch.no_grad():
            for images, masks, idx in tqdm(dataloader, desc="Predicting", total=len(dataloader)):
                images = images.to(device)
                masks  = masks.to(device).float()

                outputs = self.model(images)
                probs   = torch.sigmoid(outputs)
                preds   = (probs > threshold).float()

                images_np = images.cpu().numpy()
                masks_np  = masks.cpu().numpy()
                preds_np  = preds.cpu().numpy()
                probs_np  = probs.cpu().numpy()

                for i in range(images_np.shape[0]):
                    all_patients.append(Patient(
                        idx[i],
                        images_np[i],
                        masks_np[i],
                        preds_np[i],
                        probs_np[i],
                    ))

        self.all_patients = all_patients
        return all_patients

    # ------------------------------------------------------------------

    def show_evaluation(self):
        def fmt(x):
            return float('nan') if x is None else x

        print("=" * 50, "Evaluation Results", "=" * 50)
        print(f"Model: {self.model.__class__.__name__}")

        print("\nVoxel-wise Metrics:")
        print(f"  - Dice Score : {fmt(self.dice):.4f}")
        print(f"  - Precision  : {fmt(self.precision):.4f}")
        print(f"  - Recall     : {fmt(self.recall):.4f}")

        print("\nLesion-wise Metrics:")
        print(f"  - LW Precision       : {fmt(self.lw_precision):.4f}")
        print(f"  - LW Recall          : {fmt(self.lw_recall):.4f}")
        print(f"  - Mean   Centroid Δ  : {fmt(self.mean_center_distance):.4f}")
        print(f"  - Median Centroid Δ  : {fmt(self.median_center_distance):.4f}")

        print("\nHeatmap Quality Metrics:")
        print(f"  - ROC-AUC : {fmt(self.model_auc):.4f}")
        print(f"  - PR-AUC  : {fmt(self.pr_auc):.4f}")

    # ======================================================================
    # Loss functions
    # ======================================================================

    def combined_loss(self, outputs, masks, alpha=0.5):
        dice  = self.dice_loss(outputs, masks)
        focal = self.focal_loss(outputs, masks)
        return alpha * dice + (1 - alpha) * focal

    def dice_loss(self, logits, target, eps=1e-7):
        prob = torch.sigmoid(logits)
        intersection = (prob * target).sum()
        union        = prob.sum() + target.sum()
        return 1 - (2 * intersection + eps) / (union + eps)

    def focal_loss(self, logits, targets, alpha=0.75, gamma=2.0):
        '''
        Binary focal loss with class-specific alpha weighting.

        FIX (critical for small lesions):
          - alpha was applied uniformly to ALL voxels, which actually
            downweights the rare foreground class (lesions).
          - Correct formulation uses alpha_t:
              alpha   for foreground voxels  (targets == 1)
              1-alpha for background voxels  (targets == 0)
          - alpha raised to 0.75 to strongly upweight the rare lesion class.
            (standard value for highly imbalanced segmentation)
        '''
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs   = torch.sigmoid(logits)
        pt      = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)   # FIX
        loss    = alpha_t * (1 - pt) ** gamma * bce
        return loss.mean()

    # ======================================================================
    # Metric helpers
    # ======================================================================

    def dice_score(self, preds, masks, eps=1e-7):
        intersection = (preds * masks).sum()
        union        = preds.sum() + masks.sum()
        return (2 * intersection + eps) / (union + eps)

    def auc_score(self, probs, labels):
        return sk_auc(labels.flatten(), probs.flatten())

    def pr_auc_score(self, probs, labels):
        precision, recall, _ = precision_recall_curve(labels.flatten(), probs.flatten())
        return auc(recall, precision)


# ==========================================================================
# Patient
# ==========================================================================

class Patient:

    _id:     int
    _dwi:    np.ndarray
    _adc:    np.ndarray          # None when only one modality is present
    _labels: np.ndarray          # shape (1, D, H, W)
    _preds:  np.ndarray          # shape (1, D, H, W)
    _probs:  np.ndarray          # shape (1, D, H, W)
    _assigned_lesions:     list
    _non_assigned_lesions: list
    _TP: int
    _FP: int
    _FN: int

    def __init__(self, idx, images, labels, preds, probs):
        '''
        images : numpy array (C, D, H, W) for the single patient
        labels : numpy array (1, D, H, W)
        preds  : numpy array (1, D, H, W)  — thresholded binary predictions
        probs  : numpy array (1, D, H, W)  — sigmoid probabilities ∈ [0,1]
        '''
        self._id  = idx
        self._dwi = images[0]
        # FIX: images.shape[0] is always a positive int → always truthy.
        #      Must compare against 1 to know whether a second channel exists.
        self._adc = images[1] if images.shape[0] > 1 else None

        self._labels = labels
        self._preds  = preds
        self._probs  = probs

        self._assigned_lesions     = []
        self._non_assigned_lesions = []
        self._TP = 0
        self._FP = 0
        self._FN = 0
        self._lesions_assigned = False

    def __repr__(self):
        # FIX: calling .shape on None raised AttributeError when only one
        #      modality was loaded.
        adc_info = self._adc.shape if self._adc is not None else 'N/A'
        return (
            f"\nPatient("
            f"\n  id={self._id},"
            f"\n  dwi_shape={self._dwi.shape},"
            f"\n  adc_shape={adc_info},"
            f"\n  labels_shape={self._labels.shape},"
            f"\n  preds_shape={self._preds.shape},"
            f"\n  probs_shape={self._probs.shape},"
            f"\n  assigned_lesions={len(self._assigned_lesions)},"
            f"\n  non_assigned_lesions={len(self._non_assigned_lesions)},"
            f"\n  TP={self._TP}, FP={self._FP}, FN={self._FN}"
            f"\n)"
        )

    # ------------------------------------------------------------------
    # Lesion-wise TP / FP / FN
    # ------------------------------------------------------------------

    def find_TP_FP_FN(self):
        if not self._lesions_assigned:
            self.assign_lesions()

        self._TP = 0
        self._FP = 0
        self._FN = 0

        for gt in self._assigned_lesions:
            if gt.peers:
                self._TP += 1
            else:
                self._FN += 1

        self._FP = len(self._non_assigned_lesions)

    def lw_precision_recall(self):
        lw_precision = (self.TP / (self.TP + self.FP)
                        if (self.TP + self.FP) > 0 else float('nan'))
        lw_recall    = (self.TP / (self.TP + self.FN)
                        if (self.TP + self.FN) > 0 else float('nan'))
        return lw_precision, lw_recall

    # ------------------------------------------------------------------
    # Lesion assignment
    # ------------------------------------------------------------------

    def assign_lesions(self) -> None:
        '''
        Greedy nearest-neighbour matching between GT and predicted lesion
        centroids.  Each GT lesion is matched to the closest unowned predicted
        lesion within `_match_radius` voxels.  Predicted fragments that all
        land within radius of the same GT are merged into a single virtual
        centroid (join_centroids) so they count as one TP.

        Unmatched predicted lesions → FP.
        GT lesions with no peer    → FN.

        min_voxels_pred = 1: single-voxel blobs in the prediction are
        treated as noise and excluded from the FP count.  GT lesions are
        never filtered (even a 1-voxel GT lesion should be detected).
        '''
        centroid_list_mask, _            = self.compute_centroids_of_lesions(self.labels,
                                                                              min_voxels=0)
        centroid_list_pred, labeled_pred = self.compute_centroids_of_lesions(self.preds,
                                                                              min_voxels=1)

        # Greedy matching
        while True:
            temp_assignment = []
            for gt in centroid_list_mask:
                x = self.find_min_distance(gt, centroid_list_pred)
                if x != (None, None, None):
                    temp_assignment.append(x)
            if not temp_assignment:
                break
            temp_assignment.sort(key=lambda x: x[2])
            for gt, pred, _ in temp_assignment:
                gt.add_peer(pred)

        # Merge prediction fragments that mapped to the same GT lesion
        i = -1
        for gt in centroid_list_mask:
            if gt.peers and len(gt.peers) > 1:
                new_centroid = self.join_centroids(labeled_pred, gt.peers, i)
                gt.replace_peer(new_centroid)
            i -= 1

        self._assigned_lesions     = centroid_list_mask
        self._non_assigned_lesions = [c for c in centroid_list_pred if not c.owned]
        self._lesions_assigned     = True

    def center_distance(self) -> list:
        '''
        Returns a flat list of Euclidean centroid distances (in voxels) for
        every TP pair.  Empty list → no TP pairs found.
        Reuses the assignment cached by find_TP_FP_FN() if already done.
        '''
        if not self._lesions_assigned:
            self.assign_lesions()

        distances = []
        for gt in self.assigned_lesions:
            if gt.peers:
                pred     = gt.peers[-1]   # merged centroid, or the single match
                distance = np.linalg.norm(np.array(gt.coords) - np.array(pred.coords))
                distances.append(distance)
        return distances

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def isolate_lesions(self, img: np.ndarray) -> tuple:
        labeled_arr, n_lesions = label(img)
        return labeled_arr, n_lesions

    def compute_centroids_of_lesions(self, img: np.ndarray,
                                     min_voxels: int = 0) -> tuple:
        '''
        Returns (list[Centroid], labeled_array).

        FIX: img arrives as (1, D, H, W) — with the channel dimension.
             Passing a 4D array to scipy.ndimage.label + center_of_mass
             produced 4-element coordinate tuples; Centroid.__repr__ then
             silently dropped the W coordinate (always showed channel, D, H).
             Squeezing to 3D first gives correct (D, H, W) centroids.

        min_voxels: components with <= min_voxels voxels are discarded.
             Use min_voxels=1 for predictions (filters isolated single-voxel
             noise blobs that would otherwise inflate the FP count).
             Use min_voxels=0 for GT (never discard GT lesions, even tiny ones).
        '''
        img_3d = np.squeeze(img)          # (1,D,H,W) → (D,H,W)
        labeled_arr, n_lesions = self.isolate_lesions(img_3d)

        centroids = []
        for i in range(1, n_lesions + 1):
            component = labeled_arr == i
            if int(component.sum()) > min_voxels:
                centroid = Centroid(i, center_of_mass(component))
                centroids.append(centroid)
        return centroids, labeled_arr

    def join_centroids(self, labeled_array: np.ndarray,
                       peers: list, new_id: int) -> 'Centroid':
        '''Merge multiple predicted components into one virtual centroid.'''
        merged = np.zeros_like(labeled_array, dtype=bool)
        for peer in peers:
            merged |= (labeled_array == peer._id)
        return Centroid(new_id, center_of_mass(merged))

    def find_min_distance(self, gt: 'Centroid', centroids: list,
                          radius: float = 50.0) -> tuple:
        '''
        Returns the (gt, pred, distance) triple with the smallest distance
        among all unowned predicted centroids within `radius` voxels of `gt`.
        Returns (None, None, None) when nothing qualifies.
        '''
        eligible = []
        for cent in centroids:
            if not cent._owned:
                d = np.linalg.norm(np.array(gt.coords) - np.array(cent.coords))
                if d < radius:
                    eligible.append((gt, cent, d))
        return min(eligible, key=lambda x: x[2]) if eligible else (None, None, None)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dwi(self):
        return self._dwi

    @property
    def adc(self):
        return self._adc

    @property
    def labels(self):
        return self._labels

    @property
    def preds(self):
        return self._preds

    @property
    def probs(self):
        return self._probs

    @property
    def assigned_lesions(self):
        return self._assigned_lesions

    @property
    def non_assigned_lesions(self):
        return self._non_assigned_lesions

    @property
    def TP(self):
        return self._TP

    @property
    def FP(self):
        return self._FP

    @property
    def FN(self):
        return self._FN


# ==========================================================================
# Centroid
# ==========================================================================

class Centroid:

    _id:     int
    _coords: tuple   # always 3D: (d, h, w) after the squeeze fix
    _peers:  list
    _owned:  bool

    def __init__(self, id: int, coords: tuple):
        self._id     = id
        self._coords = coords
        self._peers  = []
        self._owned  = False

    def __repr__(self):
        c = self._coords
        sx = (f"\nCentroid(id={self._id}, "
              f"coords=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}), "
              f"owned={self._owned})")
        if self._peers:
            sx += f"  peers={[p._id for p in self._peers]}"
        return sx

    @property
    def id(self):
        return self._id

    @property
    def coords(self):
        return self._coords

    @property
    def d(self):
        return self._coords[0]

    @property
    def h(self):
        return self._coords[1]

    @property
    def w(self):
        return self._coords[2]

    @property
    def peers(self):
        return self._peers

    @property
    def owned(self):
        return self._owned

    @owned.setter
    def owned(self, value: bool):
        self._owned = value

    def add_peer(self, peer: 'Centroid'):
        if not peer._owned:
            self._peers.append(peer)
            peer._owned = True

    def replace_peer(self, new_peer: 'Centroid'):
        '''Replace all current peers with a single merged centroid.'''
        self._peers        = [new_peer]
        new_peer._owned    = True
