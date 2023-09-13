import pytest

from medem.utils import add_age


@pytest.mark.parametrize(
    ["ref_datetime", "ref_label"],
    [(2022, "age_in_2022"), ("death_datetime", "age_to_death_datetime")],
)
def test_add_age(mock_person, ref_datetime, ref_label):
    patients_w_age = add_age(
        df=mock_person,
        ref_datetime=ref_datetime,
        birth_datetime_col="birth_datetime",
        colname_age=ref_label,
    )
    assert ref_label in patients_w_age.columns
