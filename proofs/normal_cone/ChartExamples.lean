import LinearDual
namespace NormalConeExamples
open FieldDual
def chart0A : Fin 2 → Fin 3 → ℤ := ![![1, 0, 0], ![-1, 0, 0]]
def chart0b : Fin 2 → ℤ := ![0, -1]
def chart0y : Fin 2 → ℤ := ![1, 1]
theorem chart0_check : inequalityCheck chart0A chart0b chart0y = true := by decide
theorem chart0_no : ¬ ∃ x : Fin 3 → ℝ, ∀ i, ∑ j, (chart0A i j : ℝ)*x j ≤ (chart0b i : ℝ) := inequalityCheck_sound _ _ _ chart0_check
#print axioms chart0_no
def chart1A : Fin 2 → Fin 3 → ℤ := ![![-1, 0, 0], ![1, 0, 0]]
def chart1b : Fin 2 → ℤ := ![0, -1]
def chart1y : Fin 2 → ℤ := ![1, 1]
theorem chart1_check : inequalityCheck chart1A chart1b chart1y = true := by decide
theorem chart1_no : ¬ ∃ x : Fin 3 → ℝ, ∀ i, ∑ j, (chart1A i j : ℝ)*x j ≤ (chart1b i : ℝ) := inequalityCheck_sound _ _ _ chart1_check
#print axioms chart1_no
def chart2A : Fin 2 → Fin 3 → ℤ := ![![0, 1, 0], ![0, -1, 0]]
def chart2b : Fin 2 → ℤ := ![0, -1]
def chart2y : Fin 2 → ℤ := ![1, 1]
theorem chart2_check : inequalityCheck chart2A chart2b chart2y = true := by decide
theorem chart2_no : ¬ ∃ x : Fin 3 → ℝ, ∀ i, ∑ j, (chart2A i j : ℝ)*x j ≤ (chart2b i : ℝ) := inequalityCheck_sound _ _ _ chart2_check
#print axioms chart2_no
def chart3A : Fin 2 → Fin 3 → ℤ := ![![0, -1, 0], ![0, 1, 0]]
def chart3b : Fin 2 → ℤ := ![0, -1]
def chart3y : Fin 2 → ℤ := ![1, 1]
theorem chart3_check : inequalityCheck chart3A chart3b chart3y = true := by decide
theorem chart3_no : ¬ ∃ x : Fin 3 → ℝ, ∀ i, ∑ j, (chart3A i j : ℝ)*x j ≤ (chart3b i : ℝ) := inequalityCheck_sound _ _ _ chart3_check
#print axioms chart3_no
def chart4A : Fin 2 → Fin 3 → ℤ := ![![0, 0, 1], ![0, 0, -1]]
def chart4b : Fin 2 → ℤ := ![0, -1]
def chart4y : Fin 2 → ℤ := ![1, 1]
theorem chart4_check : inequalityCheck chart4A chart4b chart4y = true := by decide
theorem chart4_no : ¬ ∃ x : Fin 3 → ℝ, ∀ i, ∑ j, (chart4A i j : ℝ)*x j ≤ (chart4b i : ℝ) := inequalityCheck_sound _ _ _ chart4_check
#print axioms chart4_no
def chart5A : Fin 2 → Fin 3 → ℤ := ![![0, 0, -1], ![0, 0, 1]]
def chart5b : Fin 2 → ℤ := ![0, -1]
def chart5y : Fin 2 → ℤ := ![1, 1]
theorem chart5_check : inequalityCheck chart5A chart5b chart5y = true := by decide
theorem chart5_no : ¬ ∃ x : Fin 3 → ℝ, ∀ i, ∑ j, (chart5A i j : ℝ)*x j ≤ (chart5b i : ℝ) := inequalityCheck_sound _ _ _ chart5_check
#print axioms chart5_no

/-- Complete direction exclusion for the analytic opposite-axis cone. -/
theorem opposite_axes_only_zero (x : Fin 3 → ℝ)
    (h : ∀ j, x j ≤ 0 ∧ -x j ≤ 0) : x = 0 := by
  funext j
  have hj := h j
  change x j = 0
  linarith

theorem no_nonzero_opposite_axes :
    ¬ ∃ x : Fin 3 → ℝ, (∀ j, x j ≤ 0 ∧ -x j ≤ 0) ∧ x ≠ 0 := by
  rintro ⟨x, hx, hne⟩
  exact hne (opposite_axes_only_zero x hx)
#print axioms no_nonzero_opposite_axes
end NormalConeExamples
