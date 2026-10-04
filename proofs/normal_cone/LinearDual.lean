import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FinCases

namespace FieldDual
open Finset

def action {m n : ℕ} (A : Fin m → Fin n → ℤ) (y : Fin m → ℤ) (j : Fin n) : ℤ :=
  ∑ i, y i * A i j
def target {m : ℕ} (b y : Fin m → ℤ) : ℤ := ∑ i, y i * b i

def orthantCheck {m n : ℕ} (A : Fin m → Fin n → ℤ) (b y : Fin m → ℤ) : Bool :=
  decide ((∀ j, 0 ≤ action A y j) ∧ target b y < 0)
def inequalityCheck {m n : ℕ} (A : Fin m → Fin n → ℤ) (b y : Fin m → ℤ) : Bool :=
  decide ((∀ i, 0 ≤ y i) ∧ (∀ j, action A y j = 0) ∧ target b y < 0)

theorem action_cast {m n : ℕ} (A : Fin m → Fin n → ℤ) (y : Fin m → ℤ) (j : Fin n) :
    ((action A y j : ℤ) : ℝ) = ∑ i, (y i : ℝ) * (A i j : ℝ) := by
  simp [action]
theorem target_cast {m : ℕ} (b y : Fin m → ℤ) :
    ((target b y : ℤ) : ℝ) = ∑ i, (y i : ℝ) * (b i : ℝ) := by simp [target]

theorem weighted_identity {m n : ℕ} (A : Fin m → Fin n → ℤ) (y : Fin m → ℤ)
    (x : Fin n → ℝ) :
    (∑ i, (y i : ℝ) * (∑ j, (A i j : ℝ)*x j)) =
    ∑ j, ((action A y j : ℤ) : ℝ)*x j := by
  simp only [action_cast, Finset.mul_sum, Finset.sum_mul, mul_assoc]
  exact Finset.sum_comm

theorem orthantCheck_sound {m n : ℕ} (A : Fin m → Fin n → ℤ) (b y : Fin m → ℤ)
    (h : orthantCheck A b y = true) :
    ¬ ∃ x : Fin n → ℝ, (∀ j, 0 ≤ x j) ∧ (∀ i, ∑ j, (A i j : ℝ)*x j = (b i : ℝ)) := by
  have hd : (∀ j, 0 ≤ action A y j) ∧ target b y < 0 := of_decide_eq_true h
  rintro ⟨x,hx,he⟩
  have hn : (0 : ℝ) ≤ ∑ j, ((action A y j : ℤ) : ℝ)*x j :=
    sum_nonneg fun j _ => mul_nonneg (by exact_mod_cast hd.1 j) (hx j)
  have hid := weighted_identity A y x
  simp only [he] at hid
  rw [← target_cast] at hid
  have hc : ((target b y : ℤ) : ℝ) < 0 := by exact_mod_cast hd.2
  linarith

/-- Covers every actual RHS below b, including a whole box, with arbitrary translations. -/
theorem inequalityCheck_sound {m n : ℕ} (A : Fin m → Fin n → ℤ) (b y : Fin m → ℤ)
    (h : inequalityCheck A b y = true) :
    ¬ ∃ x : Fin n → ℝ, ∀ i, ∑ j, (A i j : ℝ)*x j ≤ (b i : ℝ) := by
  have hd : (∀ i, 0 ≤ y i) ∧ (∀ j, action A y j = 0) ∧ target b y < 0 :=
    of_decide_eq_true h
  rintro ⟨x,hx⟩
  have hs : (∑ i, (y i : ℝ)*(∑ j, (A i j : ℝ)*x j)) ≤
      ∑ i, (y i : ℝ)*(b i : ℝ) :=
    sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hx i) (by exact_mod_cast hd.1 i)
  rw [weighted_identity, ←target_cast] at hs
  simp only [hd.2.1, Int.cast_zero, zero_mul, sum_const_zero] at hs
  have hc : ((target b y : ℤ) : ℝ) < 0 := by exact_mod_cast hd.2.2
  linarith

theorem inequalityCheck_box_sound {m n : ℕ} (A : Fin m → Fin n → ℤ) (b y : Fin m → ℤ)
    (h : inequalityCheck A b y = true) (rhs : Fin m → ℝ)
    (hb : ∀ i, rhs i ≤ (b i : ℝ)) :
    ¬ ∃ x : Fin n → ℝ, ∀ i, ∑ j, (A i j : ℝ)*x j ≤ rhs i := by
  rintro ⟨x,hx⟩
  exact inequalityCheck_sound A b y h ⟨x,fun i => (hx i).trans (hb i)⟩

def soc3Check {m : ℕ} (A : Fin m → Fin 3 → ℤ) (b y : Fin m → ℤ) : Bool :=
  decide (0 ≤ action A y 0 ∧ (action A y 1)^2+(action A y 2)^2 ≤ (action A y 0)^2 ∧ target b y < 0)

theorem soc3_dot_nonnegative (a b c d e f : ℝ)
    (ha : 0 ≤ a) (hd : 0 ≤ d) (habc : b^2+c^2 ≤ a^2) (hdef : e^2+f^2 ≤ d^2) :
    0 ≤ a*d+b*e+c*f := by
  have hp := mul_nonneg ha hd
  have h1 := mul_le_mul_of_nonneg_right habc (sq_nonneg d)
  have h2 := mul_le_mul_of_nonneg_left hdef (add_nonneg (sq_nonneg b) (sq_nonneg c))
  have hcs : (b*e+c*f)^2 ≤ (a*d)^2 := by
    nlinarith [sq_nonneg (b*f-c*e)]
  by_contra h
  have hn : a*d+(b*e+c*f) < 0 := by linarith
  have hdif : 0 < (a*d-(b*e+c*f)) := by linarith
  have hh := mul_neg_of_pos_of_neg hdif hn
  nlinarith

theorem soc3Check_sound {m : ℕ} (A : Fin m → Fin 3 → ℤ) (b y : Fin m → ℤ)
    (h : soc3Check A b y = true) :
    ¬ ∃ x : Fin 3 → ℝ, (0 ≤ x 0 ∧ (x 1)^2+(x 2)^2 ≤ (x 0)^2) ∧
      (∀ i, ∑ j, (A i j : ℝ)*x j = (b i : ℝ)) := by
  have hd : 0 ≤ action A y 0 ∧ (action A y 1)^2+(action A y 2)^2 ≤ (action A y 0)^2 ∧ target b y < 0 :=
    of_decide_eq_true h
  rintro ⟨x,hx,he⟩
  have hn := soc3_dot_nonnegative ((action A y 0 : ℤ):ℝ) ((action A y 1 : ℤ):ℝ)
    ((action A y 2 : ℤ):ℝ) (x 0) (x 1) (x 2)
    (by exact_mod_cast hd.1) hx.1 (by exact_mod_cast hd.2.1) hx.2
  have hid := weighted_identity A y x
  simp only [he] at hid
  rw [←target_cast] at hid
  have hs : (∑ j : Fin 3, ((action A y j : ℤ):ℝ)*x j) =
      ((action A y 0 : ℤ):ℝ)*x 0 + ((action A y 1 : ℤ):ℝ)*x 1 + ((action A y 2 : ℤ):ℝ)*x 2 := by
    simp [Fin.sum_univ_succ, add_assoc]
  rw [hs] at hid
  have hc : ((target b y : ℤ):ℝ) < 0 := by exact_mod_cast hd.2.2
  linarith

#print axioms orthantCheck_sound
#print axioms inequalityCheck_sound
#print axioms inequalityCheck_box_sound
#print axioms soc3Check_sound
end FieldDual
